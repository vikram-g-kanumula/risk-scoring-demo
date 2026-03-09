"""
model_trainer.py  v6
====================
GLM Improver Architecture: Separate Freq×Sev GLMs (regulatory baseline)
+ M̂ ML ensemble trained on GLM residuals (Tier 3 interaction discovery)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRACK 1 — REGULATORY GLM BASELINE  (Poisson freq × Gamma sev)
  • PoissonRegressor  → λ_GLM  (annual claim frequency, T1+T2 features)
  • GammaRegressor    → μ_GLM  (expected loss given claim, T1+T2 features,
                                 trained on claimants only)
  • Baseline E[L] = λ_GLM × μ_GLM — industry gold standard (Munich Re /
    Swiss Re treaty requirement: separate freq+sev for capital allocation)
  • exp(coef) = multiplicative rate relativities, DOI-auditable
  • Tweedie GLM (p=1.65) RETAINED as display artifact for rate-filing
    narrative in UI — not used in pricing chain
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRACK 2 — M̂ INTERACTION DISCOVERY LAYER  (GLM Improver Pattern)
  • Training target: group-level O/E ratio = Sigma(actual_loss) / Sigma(GLM_pred)
    grouped by T3 interaction cells (wildfire x roof x flood x eq x slope)
  • Group O/E solves the 94%-zero-claim problem: individual M_actual = 0
    for no-claim policies; group aggregation recovers the systematic signal
  • M̂ ensemble (RF + HistGBM + ExtraTrees -> Ridge + Isotonic meta) learns
    which continuous T3 co-exposure profiles drive O/E above or below 1.0
  • M̂ normalized so portfolio mean = 1.0; M̂ > 1.0 = interaction uplift,
    M̂ < 1.0 = protective effect
  • Full pipeline: E[L] = lambda_GLM x mu_GLM x M̂
  • When M̂ = 1.0, pipeline collapses to pure GLM baseline — exact identity
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reference: Dai (CAS E-Forum, Spring 2018) — foundational GLM Improver paper
           Yan et al. (CAS E-Forum Winter 2009) — log(M̂) offset mechanism
           König & Loser (DAV, 2020–2024) — GBM for insurance pricing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Artifact keys freq_model / sev_model / xgb_m / lgb_m retained for
predictor.py backward compatibility (display-only, not in pricing chain).
Pricing chain keys: poisson_glm, gamma_glm, glm_scaler, rf_m, ridge_meta,
iso_meta, m_hat_scale_factor.
"""

import os, pickle, warnings, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                               HistGradientBoostingRegressor)
from sklearn.linear_model import (Ridge, TweedieRegressor,
                                   PoissonRegressor, GammaRegressor)
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)

# ── Feature lists ──────────────────────────────────────────────────────────────
# T12: Tier 1+2 — regulatory-safe, filed in GLM
T12 = [
    "construction_type","home_age","home_value","coverage_amount","coverage_ratio",
    "square_footage","stories","protection_class","occupancy",
    "prior_claims_3yr","credit_score","credit_restricted","deductible",
    "swimming_pool","trampoline","dog",
    "security_system","smoke_detectors","sprinkler_system","gated_community",
    "has_knob_tube_wiring","has_polybutylene_pipe",
    "permit_score","dist_to_fire_station_mi",
]
# T3: Tier 3 — compound-peril co-exposure; input to M̂ ensemble only
T3 = [
    "wildfire_zone","flood_zone","earthquake_zone","roof_material",
    "hail_zone","vegetation_risk_composite",
    "defensible_space_score",
    "dist_to_coast_mi","dist_to_fire_station_mi",
    "roof_age_yr","construction_type","state",
    "slope_steepness_pct","post_burn_rainfall_intensity",
]
SEV_FEATS = list(dict.fromkeys(T12 + T3))   # ML sev model: T1+T2+T3 (display only)
CAT_COLS  = ["construction_type","occupancy","wildfire_zone",
             "flood_zone","earthquake_zone","roof_material","state",
             "hail_zone","vegetation_risk_composite"]

# O/E clipping for M̂ target
OE_CLIP_LO   = 0.10    # floor (protective / data-sparse groups)
OE_CLIP_HI   = 8.00    # cap   (extreme compound-peril groups)
GLM_EL_FLOOR = 50.0    # minimum GLM E[L] denominator per cell (prevents /0)
MIN_GROUP_POLICIES = 10  # cells below this size default to O/E = 1.0 (neutral)

# Noise sigma for display-only ML freq/sev models
NOISE_SIGMA = 0.20

# Numeric imputation means: filled from training set, applied at inference
_IMPUTE_MEANS: dict = {}


# ── Helpers ────────────────────────────────────────────────────────────────────
def encode(df, encoders=None, fit=True):
    """Label-encode categoricals + mean-impute numeric NaNs."""
    df = df.copy()
    if encoders is None:
        encoders = {}
    num_cols = [c for c in df.columns
                if df[c].dtype in [np.float64, np.float32, float]
                and c not in CAT_COLS and df[c].isna().any()]
    for col in num_cols:
        if fit:
            _IMPUTE_MEANS[col] = float(df[col].mean())
        df[col] = df[col].fillna(_IMPUTE_MEANS.get(col, 0.0))
    for col in CAT_COLS:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le   = encoders[col]
            raw  = df[col].astype(str).values
            safe = np.where(np.isin(raw, le.classes_), raw, le.classes_[0])
            df[col] = le.transform(safe)
    return df, encoders


def gini_score(y_true, y_pred):
    """Normalized Gini. Bounded ~0.30 by 94%-zero-claim structure."""
    total = float(y_true.sum())
    if total == 0:
        return 0.0
    idx = np.argsort(y_pred)
    cl  = np.cumsum(y_true[idx]) / total
    cp  = np.arange(1, len(y_true) + 1) / len(y_true)
    _trapz = getattr(np, "trapezoid", np.trapz)  # np.trapezoid added in 1.25
    return float(np.clip(1.0 - 2.0 * _trapz(cl, cp), 0.0, 1.0))


def decile_loss_ratios(y_true, y_pred, premiums):
    """Loss ratio per predicted-risk decile — standard actuarial lift metric."""
    df = pd.DataFrame({"act": y_true, "pred": y_pred, "prem": premiums})
    df["dec"] = pd.qcut(df["pred"], 10, labels=False, duplicates="drop")
    lr = (df.groupby("dec")
            .apply(lambda g: g["act"].sum() / max(g["prem"].sum(), 1) * 100)
            .reset_index(name="loss_ratio"))
    return lr["loss_ratio"].tolist()


def reg_metrics(y_true, y_pred, label=""):
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-6))) * 100)
    print(f"   {label:52s}  R²={r2:6.3f}  MAE={mae:9.1f}  MAPE={mape:.1f}%")
    return dict(R2=round(r2,4), MAE=round(mae,2), RMSE=round(rmse,2), MAPE=round(mape,2))


def _make_cell_key(encoded_df, slope_col="slope_steepness_pct"):
    """
    Build string cell key from 5 primary T3 interaction dimensions.
    Combines encoded zone integers + binary slope bin.
    Returns pd.Series of strings like "2|4|0|1|1".
    """
    slope_bin = (encoded_df[slope_col].values > 55).astype(int)
    return (encoded_df["wildfire_zone"].astype(str)    + "|" +
            encoded_df["roof_material"].astype(str)    + "|" +
            encoded_df["flood_zone"].astype(str)       + "|" +
            encoded_df["earthquake_zone"].astype(str)  + "|" +
            pd.Series(slope_bin, index=encoded_df.index).astype(str))


def _compute_group_oe(claim_amounts, glm_el, cell_keys, min_group=10):
    """
    Compute group-level O/E = sum(actual) / sum(GLM_pred) per T3 interaction cell.

    Solves the 94%-zero-claim problem: individual M_actual = claim/pred is 0 for
    no-claim policies. Group aggregation recovers the systematic interaction signal —
    exactly how actuaries compute experience studies (O/E by rating cell).

    Groups with < min_group policies fall back to 1.0 (neutral — insufficient data).

    Returns:
        policy_oe  : np.ndarray aligned to input rows
        cell_oe_map: dict {cell_key -> O/E value}  (for lookup in val/test)
        oe_agg     : pd.DataFrame with group-level diagnostics
    """
    df_oe = pd.DataFrame({
        "claim":  np.asarray(claim_amounts, dtype=float),
        "glm_el": np.asarray(glm_el, dtype=float),
        "cell":   np.asarray(cell_keys, dtype=str),
    })
    agg = df_oe.groupby("cell").agg(
        total_claim=("claim",  "sum"),
        total_glm  =("glm_el", "sum"),
        n_policies =("claim",  "count"),
    ).reset_index()
    agg["oe_raw"] = agg["total_claim"] / agg["total_glm"].clip(lower=GLM_EL_FLOOR)
    agg["oe"]     = np.where(
        agg["n_policies"] >= min_group,
        agg["oe_raw"].clip(OE_CLIP_LO, OE_CLIP_HI),
        1.0,   # neutral fallback for sparse cells
    )
    cell_oe_map = dict(zip(agg["cell"].values, agg["oe"].values))
    policy_oe   = df_oe["cell"].map(cell_oe_map).fillna(1.0).values
    return policy_oe, cell_oe_map, agg


# ── Main trainer ───────────────────────────────────────────────────────────────
def train_all(df: pd.DataFrame) -> dict:
    t0 = time.time()
    print("=" * 72)
    print("  HOMEOWNERS RISK SCORING — TRAINING v6  (GLM Improver Architecture)")
    print("=" * 72)

    # Stratified 60/20/20 split
    tr_df, tmp   = train_test_split(df, test_size=0.40, random_state=42,
                                    stratify=df["claim_occurred"])
    va_df, te_df = train_test_split(tmp, test_size=0.50, random_state=42,
                                    stratify=tmp["claim_occurred"])
    print(f"  Train:{len(tr_df):,}  |  Val:{len(va_df):,}  |  Test:{len(te_df):,}")
    print(f"  Claim rate — train:{tr_df['claim_occurred'].mean():.2%}  "
          f"test:{te_df['claim_occurred'].mean():.2%}")

    needed = list(dict.fromkeys(SEV_FEATS + [
        "lambda_true","mu_true","M_true","expected_loss_true",
        "claim_occurred","claim_amount","risk_score_true","indicated_premium",
        "coverage_ratio","credit_restricted","permit_score","defensible_space_score",
        "slope_steepness_pct","post_burn_rainfall_intensity"]))
    tr, enc = encode(tr_df[needed], fit=True)
    va, _   = encode(va_df[needed], encoders=enc, fit=False)
    te, _   = encode(te_df[needed], encoders=enc, fit=False)
    metrics = {}

    # =========================================================================
    # TRACK 1A — POISSON GLM  (lambda_GLM: annual claim frequency)
    # =========================================================================
    print("\n" + "─"*72)
    print("  TRACK 1A — POISSON GLM  |  lambda_GLM  |  T1+T2  |  Regulatory")
    print("─"*72)
    print("  Target: claim_occurred (0/1). Industry standard for frequency modeling.")

    Xg_tr = tr[T12].values.astype(float)
    Xg_va = va[T12].values.astype(float)
    Xg_te = te[T12].values.astype(float)

    # Shared StandardScaler for both Poisson and Gamma GLMs (same T12 space)
    glm_scaler = StandardScaler()
    Xg_tr_sc   = glm_scaler.fit_transform(Xg_tr)
    Xg_va_sc   = glm_scaler.transform(Xg_va)
    Xg_te_sc   = glm_scaler.transform(Xg_te)

    y_freq_tr = tr["claim_occurred"].values.astype(float)
    y_freq_te = te["claim_occurred"].values.astype(float)

    poisson_glm = PoissonRegressor(alpha=0.01, max_iter=2000)
    poisson_glm.fit(Xg_tr_sc, y_freq_tr)

    lam_glm_tr = np.clip(poisson_glm.predict(Xg_tr_sc), 0.002, 0.30)
    lam_glm_va = np.clip(poisson_glm.predict(Xg_va_sc), 0.002, 0.30)
    lam_glm_te = np.clip(poisson_glm.predict(Xg_te_sc), 0.002, 0.30)

    poisson_coefs = {f: float(c) for f, c in zip(T12, poisson_glm.coef_)}
    poisson_relv  = {f: round(float(np.exp(c)), 4) for f, c in poisson_coefs.items()}
    poisson_gini  = gini_score(y_freq_te, lam_glm_te)

    print(f"\n  Poisson base rate:  exp({poisson_glm.intercept_:.4f}) = {np.exp(poisson_glm.intercept_):.4f}/yr")
    print(f"  Gini vs claim_occurred:  {poisson_gini:.4f}")
    print(f"  Mean lambda_GLM (test):  {lam_glm_te.mean():.4f}  "
          f"(actual rate: {y_freq_te.mean():.4f})")
    top6 = sorted(poisson_relv.items(), key=lambda x: abs(x[1]-1), reverse=True)[:6]
    print("  Top frequency relativities exp(beta):")
    for f, rv in top6:
        print(f"    {'up' if rv>1 else 'dn'} {f:32s}  x{rv:.4f}")

    metrics["poisson_glm"] = dict(
        gini=round(poisson_gini, 4),
        intercept=round(float(poisson_glm.intercept_), 4),
        n_features=len(T12),
    )

    # =========================================================================
    # TRACK 1B — GAMMA GLM  (mu_GLM: expected severity given claim)
    # =========================================================================
    print("\n" + "─"*72)
    print("  TRACK 1B — GAMMA GLM  |  mu_GLM  |  T1+T2  |  Claimants only")
    print("─"*72)
    print("  Target: claim_amount for claim_occurred=1 (removes zero-inflation)")

    claimant_mask_tr = y_freq_tr > 0
    y_sev_tr_claims  = tr["claim_amount"].values[claimant_mask_tr].astype(float)
    Xg_claimants_sc  = Xg_tr_sc[claimant_mask_tr]
    print(f"  Claimants in training: {claimant_mask_tr.sum():,} "
          f"({claimant_mask_tr.mean():.2%})")

    gamma_glm = GammaRegressor(alpha=0.01, max_iter=2000)
    gamma_glm.fit(Xg_claimants_sc, y_sev_tr_claims)

    # Predict for ALL policies (conditional E[severity | claim occurs])
    mu_glm_tr = np.clip(gamma_glm.predict(Xg_tr_sc), 500.0, 600_000.0)
    mu_glm_va = np.clip(gamma_glm.predict(Xg_va_sc), 500.0, 600_000.0)
    mu_glm_te = np.clip(gamma_glm.predict(Xg_te_sc), 500.0, 600_000.0)

    gamma_coefs = {f: float(c) for f, c in zip(T12, gamma_glm.coef_)}
    gamma_relv  = {f: round(float(np.exp(c)), 4) for f, c in gamma_coefs.items()}

    claimant_mask_te = y_freq_te > 0
    gamma_gini = gini_score(
        te["claim_amount"].values[claimant_mask_te],
        mu_glm_te[claimant_mask_te]
    ) if claimant_mask_te.sum() > 0 else 0.0

    print(f"\n  Gamma base severity: exp({gamma_glm.intercept_:.4f}) = "
          f"${np.exp(gamma_glm.intercept_):,.0f}")
    print(f"  Gini vs claim_amount (claimants): {gamma_gini:.4f}")
    print(f"  Mean mu_GLM (test, all policies): ${mu_glm_te.mean():,.0f}")
    top6 = sorted(gamma_relv.items(), key=lambda x: abs(x[1]-1), reverse=True)[:6]
    print("  Top severity relativities exp(beta):")
    for f, rv in top6:
        print(f"    {'up' if rv>1 else 'dn'} {f:32s}  x{rv:.4f}")

    metrics["gamma_glm"] = dict(
        gini_claimants=round(gamma_gini, 4),
        intercept=round(float(gamma_glm.intercept_), 4),
        n_features=len(T12),
        n_claimants_train=int(claimant_mask_tr.sum()),
    )

    # GLM Baseline E[L] = lambda_GLM x mu_GLM
    glm_el_tr = lam_glm_tr * mu_glm_tr
    glm_el_va = lam_glm_va * mu_glm_va
    glm_el_te = lam_glm_te * mu_glm_te

    el_true_te = te["expected_loss_true"].values
    r2_glm_vs_oracle = r2_score(el_true_te, glm_el_te)
    gini_glm_el = gini_score(te["claim_amount"].values.astype(float), glm_el_te)
    prem_te = te_df["indicated_premium"].values

    print(f"\n  GLM Baseline E[L] = lambda_GLM x mu_GLM:")
    print(f"    Mean E[L] (test):         ${glm_el_te.mean():,.0f}")
    print(f"    R² vs oracle E[L]:        {r2_glm_vs_oracle:.4f}")
    print(f"    Gini vs actual claim_amt: {gini_glm_el:.4f}  (ceiling ~0.30 at 94% zeros)")

    metrics["glm_baseline"] = dict(
        r2_vs_oracle=round(r2_glm_vs_oracle, 4),
        gini=round(gini_glm_el, 4),
        mean_el_test=round(float(glm_el_te.mean()), 2),
    )
    try:
        metrics["glm_baseline"]["decile_lr"] = [
            round(v,2) for v in decile_loss_ratios(
                te["claim_amount"].values.astype(float), glm_el_te, prem_te)]
    except Exception:
        pass

    # =========================================================================
    # TRACK 1C — TWEEDIE GLM  (Display artifact — rate-filing narrative only)
    # =========================================================================
    print("\n" + "─"*72)
    print("  TRACK 1C — TWEEDIE GLM  |  Display only  |  Rate-Filing Narrative")
    print("─"*72)

    y_tw_tr = tr["claim_amount"].values.astype(float)
    tweedie_glm = TweedieRegressor(power=1.65, alpha=0.01, max_iter=2000, link="log")
    tweedie_glm.fit(Xg_tr_sc, y_tw_tr)
    tweedie_coefs = {f: float(c) for f, c in zip(T12, tweedie_glm.coef_)}
    tweedie_relv  = {f: round(float(np.exp(c)), 4) for f, c in tweedie_coefs.items()}
    glm_relv      = tweedie_relv   # backward-compat key for app.py

    print(f"  Tweedie(p=1.65): ${np.exp(tweedie_glm.intercept_):,.0f} base rate "
          f"[NOT in pricing chain — UI display only]")

    # Keep glm metrics key for backward compat with app.py metric lookups
    metrics["glm"] = dict(
        gini=round(gini_glm_el, 4),
        r2_vs_oracle=round(r2_glm_vs_oracle, 4),
        power=1.65, alpha=0.01,
        intercept=round(float(tweedie_glm.intercept_), 4),
        n_features=len(T12),
        note="Tweedie display artifact. Pricing uses Poisson x Gamma GLMs.",
    )

    # =========================================================================
    # TRACK 2 — M̂ ENSEMBLE  (Tier 3 Interaction Discovery, GLM Improver)
    # =========================================================================
    print("\n" + "─"*72)
    print("  TRACK 2 — M̂ ENSEMBLE  |  T3 Features  |  GLM O/E Residual Target")
    print("─"*72)
    print("  M̂ target: group O/E = sum(actual_loss) / sum(lambda_GLM x mu_GLM)")
    print("  Groups: wildfire_zone x roof_material x flood_zone x eq_zone x slope_bin")
    print("  Group O/E solves 94%-zero-claim problem via actuarial aggregation")

    # Build T3 cell keys
    cell_keys_tr = _make_cell_key(tr, "slope_steepness_pct").values
    cell_keys_va = _make_cell_key(va, "slope_steepness_pct").values
    cell_keys_te = _make_cell_key(te, "slope_steepness_pct").values

    # Compute group O/E on TRAINING set
    M_actual_tr, cell_oe_map, oe_agg = _compute_group_oe(
        tr["claim_amount"].values.astype(float),
        glm_el_tr,
        cell_keys_tr,
        min_group=MIN_GROUP_POLICIES,
    )

    # Val: use training cell O/E (held out — no leakage)
    M_actual_va = pd.Series(cell_keys_va).map(cell_oe_map).fillna(1.0).values

    # Test evaluation: compute test-set O/E separately (honest out-of-sample)
    M_actual_te_eval, _, _ = _compute_group_oe(
        te["claim_amount"].values.astype(float),
        glm_el_te,
        cell_keys_te,
        min_group=5,
    )

    n_cells = len(cell_oe_map)
    oe_vals  = np.array(list(cell_oe_map.values()))
    print(f"\n  O/E cells: {n_cells} unique T3 interaction groups in training set")
    print(f"  O/E distribution across cells:")
    for p in [0, 25, 50, 75, 90, 95, 99, 100]:
        print(f"    P{p:3d}: {np.percentile(oe_vals, p):.3f}x")

    oe_overall_tr = (tr["claim_amount"].values.sum() / max(glm_el_tr.sum(), 1.0))
    print(f"\n  Overall O/E (train): {oe_overall_tr:.4f}  (1.00 = GLM perfectly calibrated)")
    print(f"  Mean M_actual (policy-level, training): {M_actual_tr.mean():.4f}")

    # Normalize: scale M̂ so portfolio mean = 1.0
    # This ensures GLM x M̂ is calibrated — M̂ redistributes, not inflates
    m_hat_scale_factor = float(M_actual_tr.mean())
    if m_hat_scale_factor < 0.05:
        m_hat_scale_factor = 1.0   # safety fallback
    M_norm_tr = (M_actual_tr / m_hat_scale_factor).clip(OE_CLIP_LO, OE_CLIP_HI)
    M_norm_va = (M_actual_va / m_hat_scale_factor).clip(OE_CLIP_LO, OE_CLIP_HI)

    print(f"\n  M̂ scale factor (normalising to mean=1.0): {m_hat_scale_factor:.4f}")
    print(f"  M_norm training distribution (post-scale):")
    for p in [0, 25, 50, 75, 90, 95, 99]:
        print(f"    P{p:3d}: {np.percentile(M_norm_tr, p):.3f}x")

    # M̂ stacked ensemble: RF + HistGBM + ExtraTrees -> Ridge + Isotonic
    print(f"\n  [2] M̂ — RF + HistGBM + ExtraTrees -> Ridge + Isotonic  (OOF stacking)")

    Xm_tr  = tr[T3].values
    Xm_va  = va[T3].values
    Xm_te  = te[T3].values
    Xm_all = np.vstack([Xm_tr, Xm_va])
    M_all  = np.concatenate([M_norm_tr, M_norm_va])

    rf_m = RandomForestRegressor(n_estimators=300, max_depth=7,
                                  min_samples_leaf=30, n_jobs=1, random_state=42)
    rf_m.fit(Xm_tr, M_norm_tr)

    hgb_m = HistGradientBoostingRegressor(
        max_iter=400, max_depth=5, learning_rate=0.04,
        min_samples_leaf=30, l2_regularization=1.0, random_state=42)
    hgb_m.fit(Xm_tr, M_norm_tr)

    et_m = ExtraTreesRegressor(n_estimators=300, max_depth=7,
                                min_samples_leaf=30, n_jobs=1, random_state=42)
    et_m.fit(Xm_tr, M_norm_tr)

    # OOF stacking for meta-learner calibration
    oof = np.zeros((len(Xm_all), 3))
    kf  = KFold(n_splits=5, shuffle=True, random_state=42)
    for fi, (idx_tr, idx_va) in enumerate(kf.split(Xm_all)):
        _rf = RandomForestRegressor(n_estimators=150, max_depth=7,
                                    min_samples_leaf=30, n_jobs=1, random_state=fi)
        _rf.fit(Xm_all[idx_tr], M_all[idx_tr])
        oof[idx_va, 0] = _rf.predict(Xm_all[idx_va])

        _h = HistGradientBoostingRegressor(max_iter=200, max_depth=4,
                                            learning_rate=0.06, random_state=fi)
        _h.fit(Xm_all[idx_tr], M_all[idx_tr])
        oof[idx_va, 1] = _h.predict(Xm_all[idx_va])

        _et = ExtraTreesRegressor(n_estimators=150, max_depth=7,
                                   min_samples_leaf=30, n_jobs=1, random_state=fi)
        _et.fit(Xm_all[idx_tr], M_all[idx_tr])
        oof[idx_va, 2] = _et.predict(Xm_all[idx_va])

    ridge_m = Ridge(alpha=1.0)
    ridge_m.fit(oof, M_all)
    iso_m = IsotonicRegression(out_of_bounds="clip")
    iso_m.fit(ridge_m.predict(oof), M_all)

    # M̂ test evaluation
    meta_te  = np.column_stack([rf_m.predict(Xm_te),
                                 hgb_m.predict(Xm_te),
                                 et_m.predict(Xm_te)])
    m_hat_te = np.clip(iso_m.predict(ridge_m.predict(meta_te)), OE_CLIP_LO, OE_CLIP_HI)

    # Evaluate vs test-set O/E (honest out-of-sample)
    M_te_norm = (M_actual_te_eval / m_hat_scale_factor).clip(OE_CLIP_LO, OE_CLIP_HI)
    metrics["m_hat"] = reg_metrics(M_te_norm, m_hat_te,
                                    "M̂ (R² vs test group O/E)")

    # Oracle correlation: M̂ vs M_true (ground-truth interaction multipliers)
    m_hat_vs_oracle_corr = float(np.corrcoef(m_hat_te, te["M_true"].values)[0,1])
    print(f"   M̂ vs oracle M_true corr (test):  {m_hat_vs_oracle_corr:.4f}  "
          f"(expected 0.70-0.90)")
    metrics["m_hat"]["oracle_corr"] = round(m_hat_vs_oracle_corr, 4)

    # =========================================================================
    # TRACK 2 (DISPLAY ONLY) — ML Freq + Sev  (Score A2 UI decomposition)
    # =========================================================================
    # These HistGBM models are shown in the Score A2 sub-score display and SHAP
    # waterfall. They are NOT in the pricing chain.
    print("\n" + "─"*72)
    print("  ML FREQ+SEV  |  Display-only  |  Score A2 + SHAP in app.py")
    print("─"*72)

    RNG_TR = np.random.default_rng(42)
    RNG_TE = np.random.default_rng(99)

    Xf_tr = tr[T12].values
    Xf_te = te[T12].values
    lam_n_tr = (tr["lambda_true"].values *
                np.exp(RNG_TR.normal(0, NOISE_SIGMA, len(tr)))).clip(0.005, 0.18)
    lam_n_te = (te["lambda_true"].values *
                np.exp(RNG_TE.normal(0, NOISE_SIGMA, len(te)))).clip(0.005, 0.18)
    freq_m = HistGradientBoostingRegressor(
        max_iter=500, max_depth=5, learning_rate=0.03,
        min_samples_leaf=30, l2_regularization=1.5, random_state=42)
    freq_m.fit(Xf_tr, lam_n_tr)
    lam_pred_disp = np.clip(freq_m.predict(Xf_te), 0.005, 0.15)
    metrics["frequency"] = reg_metrics(lam_n_te, lam_pred_disp,
                                        "lambda ML display (R² vs noisy)")

    Xs_tr = tr[SEV_FEATS].values
    Xs_te = te[SEV_FEATS].values
    mu_n_tr = (tr["mu_true"].values *
               np.exp(RNG_TR.normal(0, NOISE_SIGMA, len(tr)))).clip(1_000, 600_000)
    mu_n_te = (te["mu_true"].values *
               np.exp(RNG_TE.normal(0, NOISE_SIGMA, len(te)))).clip(1_000, 600_000)
    sev_m = HistGradientBoostingRegressor(
        max_iter=500, max_depth=5, learning_rate=0.03,
        min_samples_leaf=25, l2_regularization=1.5, random_state=42)
    sev_m.fit(Xs_tr, mu_n_tr)
    mu_pred_disp = np.clip(sev_m.predict(Xs_te), 1_000, 500_000)
    metrics["severity"] = reg_metrics(mu_n_te, mu_pred_disp,
                                       "mu ML display (R² vs noisy)")

    # =========================================================================
    # PIPELINE IMPACT ANALYSIS  (lambda_GLM x mu_GLM x M̂ — full architecture)
    # =========================================================================
    print("\n" + "─"*72)
    print("  PIPELINE IMPACT — GLM Baseline vs Full (lambda_GLM x mu_GLM x M̂)")
    print("─"*72)

    el_baseline = glm_el_te            # lambda_GLM x mu_GLM, M̂ = 1.0
    el_full     = glm_el_te * m_hat_te  # full pipeline
    act         = te["claim_amount"].values.astype(float)

    print(f"\n  M̂ distribution across {len(m_hat_te):,} test policies (post-norm):")
    for p in [0, 25, 50, 75, 90, 95, 99, 100]:
        print(f"    P{p:3d}: {float(np.percentile(m_hat_te, p)):.3f}x")

    protective = float((m_hat_te < 0.95).mean())
    no_impact  = float(((m_hat_te >= 0.95) & (m_hat_te < 1.05)).mean())
    mild       = float(((m_hat_te >= 1.05) & (m_hat_te < 1.30)).mean())
    moderate   = float(((m_hat_te >= 1.30) & (m_hat_te < 2.00)).mean())
    severe     = float((m_hat_te >= 2.00).mean())
    print(f"\n  M̂ impact tiers:")
    print(f"    Protective   (M̂ < 0.95):    {protective:.1%}  — GLM over-prices these")
    print(f"    Neutral  (0.95-1.05):    {no_impact:.1%}")
    print(f"    Mild     (1.05-1.30):    {mild:.1%}")
    print(f"    Moderate (1.30-2.00):    {moderate:.1%}")
    print(f"    Severe       (>=2.00):   {severe:.1%}  <- compounding peril co-exposure")

    prem_uplift = (el_full - el_baseline) / (el_baseline + 1)
    print(f"\n  Premium uplift from M̂: mean={prem_uplift.mean()*100:+.1f}%  "
          f"p95={float(np.percentile(prem_uplift,95))*100:+.1f}%")

    gini_full = gini_score(act, el_full)
    gini_base = gini_score(act, el_baseline)
    print(f"\n  Gini on actual claim_amount:")
    print(f"    GLM Baseline: {gini_base:.4f}  |  Full (x M̂): {gini_full:.4f}  "
          f"| Delta: {gini_full-gini_base:+.4f}")

    # Score normalization anchored to test-set predicted E[L] range
    el_min = float(np.percentile(el_baseline, 1))
    el_max = float(np.percentile(el_full, 99))

    def _score(el):
        return np.clip(
            50 + 900 * (np.log1p(el) - np.log1p(el_min)) /
                       (np.log1p(el_max) - np.log1p(el_min)),
            50, 950)

    def _band(s):
        if s < 200: return "Very Low"
        if s < 400: return "Low"
        if s < 600: return "Moderate"
        if s < 800: return "High"
        return "Very High"

    sc_baseline = _score(el_baseline)
    sc_full     = _score(el_full)
    sc_true     = te["risk_score_true"].values
    bands_base  = np.array([_band(s) for s in sc_baseline])
    bands_full  = np.array([_band(s) for s in sc_full])
    reclassed   = float((bands_full != bands_base).mean())
    upgraded    = float(((sc_full > sc_baseline) & (bands_full != bands_base)).mean())
    downgraded  = float(((sc_full < sc_baseline) & (bands_full != bands_base)).mean())

    print(f"\n  Reclassification (GLM baseline -> full pipeline):")
    print(f"    Total: {reclassed:.1%}  |  Upgraded: {upgraded:.1%}  |  Downgraded: {downgraded:.1%}")

    metrics["m_hat_distribution"] = dict(
        p0  =round(float(np.percentile(m_hat_te,  0)), 3),
        p25 =round(float(np.percentile(m_hat_te, 25)), 3),
        p50 =round(float(np.percentile(m_hat_te, 50)), 3),
        p75 =round(float(np.percentile(m_hat_te, 75)), 3),
        p90 =round(float(np.percentile(m_hat_te, 90)), 3),
        p95 =round(float(np.percentile(m_hat_te, 95)), 3),
        p99 =round(float(np.percentile(m_hat_te, 99)), 3),
        mean         =round(float(m_hat_te.mean()), 3),
        protective_pct=round(protective, 4),
        no_impact_pct =round(no_impact,  4),
        mild_pct      =round(mild,       4),
        moderate_pct  =round(moderate,   4),
        severe_pct    =round(severe,     4),
    )
    metrics["m_hat_uplift"] = dict(
        mean_pct  =round(float(prem_uplift.mean()*100), 2),
        median_pct=round(float(np.median(prem_uplift)*100), 2),
        p95_pct   =round(float(np.percentile(prem_uplift, 95)*100), 2),
        max_pct   =round(float(prem_uplift.max()*100), 2),
    )
    metrics["gini"] = dict(
        baseline=round(gini_base, 4),
        full    =round(gini_full, 4),
        delta   =round(gini_full - gini_base, 4),
        note    ="Bounded ~0.30 by 94% zero-claim rate — directional metric",
    )
    metrics["reclassification_pct"] = round(reclassed, 4)
    metrics["upgraded_pct"]         = round(upgraded, 4)
    metrics["downgraded_pct"]       = round(downgraded, 4)

    try:
        metrics["full_decile_lr"]     = [round(v,2) for v in
            decile_loss_ratios(act, el_full, prem_te)]
        metrics["baseline_decile_lr"] = [round(v,2) for v in
            decile_loss_ratios(act, el_baseline, prem_te)]
    except Exception:
        pass

    metrics["risk_score"] = reg_metrics(sc_true, sc_full,
                                         "Risk Score A1 (full vs true)")
    print(f"\n  Total training time: {time.time()-t0:.1f}s")

    # ── Persist artifacts ──────────────────────────────────────────────────────
    arts = dict(
        # === PRICING CHAIN: Poisson x Gamma GLM baseline =====================
        poisson_glm          = poisson_glm,
        gamma_glm            = gamma_glm,
        glm_scaler           = glm_scaler,      # shared scaler for both GLMs
        poisson_coefs        = poisson_coefs,
        gamma_coefs          = gamma_coefs,
        poisson_relativities = poisson_relv,
        gamma_relativities   = gamma_relv,
        glm_el_train_mean    = round(float(glm_el_tr.mean()), 2),

        # === PRICING CHAIN: M̂ ensemble =======================================
        rf_m               = rf_m,
        xgb_m              = hgb_m,          # backward-compat key (HistGBM)
        lgb_m              = et_m,           # backward-compat key (ExtraTrees)
        ridge_meta         = ridge_m,
        iso_meta           = iso_m,
        m_hat_scale_factor = m_hat_scale_factor,
        oe_clip_lo         = OE_CLIP_LO,
        oe_clip_hi         = OE_CLIP_HI,

        # === DISPLAY ONLY: Tweedie GLM (rate-filing UI narrative) =============
        glm                = tweedie_glm,    # backward-compat key
        glm_coefs          = tweedie_coefs,
        glm_relativities   = glm_relv,       # backward-compat key

        # === DISPLAY ONLY: ML freq/sev (Score A2 + SHAP in app.py) ===========
        freq_model         = freq_m,
        sev_model          = sev_m,

        # === METADATA =========================================================
        encoders           = enc,
        t12                = T12,
        t3                 = T3,
        sev_feats          = SEV_FEATS,
        cat_cols           = CAT_COLS,
        impute_means       = _IMPUTE_MEANS,
        metrics            = metrics,
        el_min             = el_min,
        el_max             = el_max,
        noise_sigma        = NOISE_SIGMA,
        n_train            = len(tr_df),
        n_val              = len(va_df),
        n_test             = len(te_df),
    )
    with open("models/artifacts.pkl", "wb") as f:
        pickle.dump(arts, f)

    te_df.to_csv("data/test_data.csv",  index=False)
    tr_df.to_csv("data/train_data.csv", index=False)
    print("✓  models/artifacts.pkl  data/test_data.csv  data/train_data.csv")
    print("=" * 72)
    return arts


if __name__ == "__main__":
    df = pd.read_csv("data/homeowners_data.csv")
    train_all(df)