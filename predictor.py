"""
predictor.py  v7
================
GLM Improver Architecture: E[L] = lambda_GLM x mu_GLM x M̂

PRICING CHAIN (what produces premiums and risk scores):
  _glm_baseline()     → lambda_GLM (PoissonRegressor) x mu_GLM (GammaRegressor)
                         Baseline E[L] — the regulatory-ready additive model
  predict_baseline()  → GLM baseline, M̂ = 1.0  (the pure Poisson x Gamma counterfactual)
  predict()           → GLM baseline x M̂        (Tier 3 interaction discovery applied)
  When M̂ = 1.0, predict() collapses exactly to predict_baseline() — mathematical identity

DISPLAY-ONLY (Score A2 sub-scores, SHAP waterfall — not in pricing chain):
  _freq_sev_display() → ML HistGBM lambda + mu   (visual decomposition for Score A2)
  get_shap_values()   → TreeExplainer SHAP on ML freq, sev, M̂ models

ARCHITECTURE NOTE — Tweedie GLM:
  The TweedieRegressor (p=1.65) stored in arts["glm"] is a display-only artifact.
  It provides the single-model exp(β) coefficient table shown in the methodology tab
  for the rate-filing narrative. It is NOT used in predict() or predict_baseline().
  Pricing baseline is always Poisson x Gamma, per Munich Re / Swiss Re treaty standard.

INTERACTION OVERRIDES:
  The m_overrides dict in DEFAULT_PRICING_CFG is DISPLAY-ONLY — used by
  _get_interactions() to show narrative labels and reference multiplier values for
  the UI. These values reflect what the M̂ ensemble has learned from GLM residuals.
  They are NOT applied to M̂ in predict() — the ensemble prediction is authoritative.

Public API (all app.py entry points):
  predict_baseline()      → baseline result dict (M̂ = 1.0, GLM only)
  predict()               → full result dict     (GLM x M̂)
  predict_both()          → (baseline, full)     for side-by-side display
  predict_whatif()        → what-if on full pipeline
  compute_tier2_only_score() → alias for predict_baseline()
  batch_predict()         → DataFrame for portfolio scatter / tab 4
  get_glm_relativities()  → {poisson, gamma, tweedie} relativities for UI
  get_shap_values()       → SHAP per model component for tab 3
"""

import pickle, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_arts = None

DEFAULT_PRICING_CFG = dict(
    # Blueprint: indicated_premium = E[L] / (1 - expense_ratio - profit_margin)
    # = E[L] / (1 - 0.28 - 0.05) = E[L] / 0.67
    target_lr    = 0.67,
    expense_load = 0.0,     # already baked into 0.67 divisor
    score_wf     = 0.45,
    score_ws     = 0.55,
    lam_cap      = 0.30,    # Poisson GLM upper clip (wider than old ML cap of 0.15)
    sev_cap      = 600_000,
    # ── DISPLAY-ONLY reference multipliers for _get_interactions() ─────────────
    # These are NOT applied in predict() — they are narrative labels reflecting
    # what the M̂ ensemble learns from GLM residuals. Grounded in AIR/RMS/ISO data.
    m_overrides  = dict(
        wood_wf_high       = 3.50,   # Wood Shake × High Wildfire
        wood_wf_mod        = 2.10,   # Wood Shake × Moderate Wildfire
        flood_coast        = 2.20,   # High Flood × <5mi coast
        nonwood_wf         = 1.80,   # Non-wood roof × High Wildfire
        eq_high            = 1.50,   # High earthquake zone
        old_frame          = 1.35,   # Old roof × Frame construction
        metal_wf_sprinkler = 0.82,   # Metal + sprinkler × WF (protective)
        defensible_high    = 0.88,   # Defensible space ≥80 in WUI (protective)
        new_superior       = 0.85,   # New Superior construction (protective)
        frame_pc_high      = 1.18,   # Frame × PC≥8
        knob_tube_fire     = 1.20,   # Knob-and-tube wiring
        polybutylene_water = 1.22,   # Polybutylene pipe
        slope_burn_rain    = 1.45,   # Slope >55% × High WF × Post-burn Rain >60
        slope_burn_only    = 1.18,   # Slope >55% × High WF (no rain)
    ),
)


def load_arts():
    global _arts
    if _arts is None:
        with open("models/artifacts.pkl", "rb") as f:
            _arts = pickle.load(f)
    return _arts


def validate_inputs(d: dict) -> list:
    out = []
    hv  = d.get("home_value", 0)
    ca  = d.get("coverage_amount", 0)
    yb  = d.get("year_built", 2000)
    ded = d.get("deductible", 1000)
    pc  = d.get("protection_class", 5)
    if ca < hv * 0.70:
        out.append(f"⚠️ Coverage (${ca:,.0f}) is below 70% of home value — underinsurance risk")
    if ca > hv * 1.30:
        out.append(f"⚠️ Coverage (${ca:,.0f}) exceeds 130% of home value — over-insurance / moral hazard signal")
    if yb > 2024:
        out.append("⚠️ Year built is in the future")
    if yb < 1900:
        out.append("⚠️ Year built before 1900 — unusual for active policy")
    if ded > hv * 0.06:
        out.append(f"⚠️ Deductible (${ded:,}) exceeds 6% of home value — high self-retention")
    if pc == 10 and d.get("dist_to_fire_station_mi", 3) < 1:
        out.append("⚠️ PC=10 but fire station distance < 1mi — data inconsistency")
    return out


def _encode_row(d: dict) -> pd.DataFrame:
    """
    Encode a single-row input dict into a DataFrame ready for model inference.
    Missing columns are filled with training imputed means so old callers don't break.
    """
    arts     = load_arts()
    enc      = arts["encoders"]
    cat_cols = arts["cat_cols"]
    imp      = arts.get("impute_means", {})

    row = pd.DataFrame([d])

    # Fill any columns expected by model that caller didn't supply
    all_feats = list(dict.fromkeys(arts["t12"] + arts["t3"]))
    for col in all_feats:
        if col not in row.columns:
            row[col] = imp.get(col, 0.0)

    # Numeric NaN → imputed mean
    for col, fill in imp.items():
        if col in row.columns:
            row[col] = pd.to_numeric(row[col], errors="coerce").fillna(fill)

    # Derived fields
    if "coverage_ratio" not in d and "coverage_amount" in d and "home_value" in d:
        hv = float(d.get("home_value", 1))
        cv = float(d.get("coverage_amount", hv))
        row["coverage_ratio"] = round(cv / hv if hv > 0 else 1.0, 3)

    if "credit_restricted" not in d and "state" in d:
        restricted_states = {"CA", "MA", "HI", "MD", "MI"}
        row["credit_restricted"] = int(str(d.get("state", "")) in restricted_states)

    # Categorical label encoding
    for col in cat_cols:
        if col not in row.columns:
            continue
        le  = enc[col]
        val = str(row[col].iloc[0])
        if val not in le.classes_:
            val = le.classes_[0]
        row[col] = le.transform([val])

    return row


def _risk_band(score):
    if score < 200: return "Very Low",  "#059669"
    if score < 400: return "Low",       "#16A34A"
    if score < 600: return "Moderate",  "#CA8A04"
    if score < 800: return "High",      "#DC6803"
    return               "Very High",   "#DC2626"


def _uw_action(band):
    return {
        "Very Low":  ("✅ Accept — Best Terms",             "#059669"),
        "Low":       ("✅ Accept — Standard Terms",         "#16A34A"),
        "Moderate":  ("🔶 Accept with Conditions",         "#CA8A04"),
        "High":      ("🔴 Refer to Senior Underwriter",    "#DC6803"),
        "Very High": ("🚫 Decline / Surplus Lines Market", "#DC2626"),
    }[band]


def _get_interactions(inp: dict, m_overrides: dict = None) -> list:
    """
    Return list of (label, ref_multiplier, color, tooltip) tuples for active T3
    interaction patterns on this property.

    IMPORTANT: The ref_multiplier values are DISPLAY-ONLY reference signals
    derived from AIR/RMS/ISO literature and calibrated to GLM O/E ratios.
    They are NOT directly applied to M̂ in the pricing chain — the M̂ ensemble
    prediction is the authoritative value. These labels explain WHAT the ensemble
    has learned from GLM residuals for this property's T3 co-exposure profile.
    Colors: #DC2626 = severe upward  |  #DC6803 = moderate upward
            #CA8A04 = mild upward    |  #059669 = protective (downward)
    """
    if m_overrides is None:
        m_overrides = DEFAULT_PRICING_CFG["m_overrides"]

    out    = []
    wood   = inp.get("roof_material") == "Wood Shake"
    metal  = inp.get("roof_material") == "Metal"
    wf     = inp.get("wildfire_zone", "Low")
    fl     = inp.get("flood_zone", "Low")
    eq     = inp.get("earthquake_zone", "Low")
    coast  = inp.get("dist_to_coast_mi", 99) < 5
    old_r  = inp.get("roof_age_yr", 0) > 20
    new_r  = inp.get("roof_age_yr", 99) <= 5
    frame  = inp.get("construction_type") == "Frame"
    sup    = inp.get("construction_type") == "Superior"
    hail   = inp.get("hail_zone", "Low")
    can    = inp.get("vegetation_risk_composite", "Low")
    prior  = inp.get("prior_claims_3yr", 0)
    spk    = bool(inp.get("sprinkler_system", 0))
    ds     = float(inp.get("defensible_space_score") or 0)
    pc     = int(inp.get("protection_class", 5))
    knob   = bool(inp.get("has_knob_tube_wiring", 0))
    poly   = bool(inp.get("has_polybutylene_pipe", 0))
    slope  = float(inp.get("slope_steepness_pct", 0))
    pbr    = float(inp.get("post_burn_rainfall_intensity", 0))

    # ── HAZARD AMPLIFIERS ──────────────────────────────────────────────────────
    if   wood and wf == "High":
        out.append(("Wood Shake × High Wildfire",      m_overrides["wood_wf_high"], "#DC2626", ""))
    elif wood and wf == "Moderate":
        out.append(("Wood Shake × Mod Wildfire",       m_overrides["wood_wf_mod"],  "#DC6803", ""))
    elif wf == "High":
        out.append(("Non-Wood × High Wildfire",        m_overrides["nonwood_wf"],   "#DC6803", ""))
    elif wood:
        out.append(("Wood Shake (base fire exposure)", 1.40,                        "#CA8A04", ""))

    if   fl == "High" and coast:
        out.append(("High Flood × Coastal Surge <5mi", m_overrides["flood_coast"],  "#DC2626", ""))
    elif fl == "High":
        out.append(("High Flood Zone",                 1.60,                        "#DC6803", ""))
    elif fl == "Moderate":
        out.append(("Moderate Flood Zone",             1.20,                        "#CA8A04", ""))

    if   eq == "High":
        out.append(("High Earthquake Zone",            m_overrides["eq_high"],      "#DC6803", ""))
    elif eq == "Moderate":
        out.append(("Moderate Earthquake Zone",        1.15,                        "#CA8A04", ""))

    if   old_r and frame:
        out.append(("Old Roof (>20yr) × Frame",        m_overrides["old_frame"],    "#DC6803", ""))
    elif old_r:
        out.append(("Aged Roof > 20 years",            1.15,                        "#CA8A04", ""))

    if   old_r and hail == "High":
        out.append(("Old Roof × High Hail Zone",       1.30,                        "#DC6803", ""))
    elif old_r and hail == "Moderate":
        out.append(("Old Roof × Mod Hail Zone",        1.20,                        "#DC6803", ""))

    if   prior >= 2 and can == "High":
        out.append(("Water Claims × High Vegetation Density",
                    1.25, "#DC6803",
                    "Dense root systems seek moisture in soil near foundation. "
                    "Properties with 2+ recent water claims and >60% canopy coverage "
                    "face a self-reinforcing damage cycle: roots penetrate pipes and "
                    "foundations, moisture retention accelerates mold growth, and "
                    "subsequent claims escalate in severity. NDVI + NDWI satellite "
                    "data confirms this moisture-retention interaction."))
    elif prior >= 2 and can == "Moderate":
        out.append(("Water Claims × Moderate Vegetation", 1.15, "#CA8A04", ""))

    # ── INTERIOR CONDITION HAZARDS ─────────────────────────────────────────────
    if knob:
        out.append(("Knob-and-Tube Wiring (interior)",
                    m_overrides.get("knob_tube_fire", 1.20), "#DC6803",
                    "Pre-1950 knob-and-tube wiring lacks grounding and modern "
                    "insulation — arc-fault ignition risk is 2–3× higher than "
                    "modern wiring. No exterior data source captures this. "
                    "Interior inspection required."))
    if poly:
        out.append(("Polybutylene Plumbing (interior)",
                    m_overrides.get("polybutylene_water", 1.22), "#DC2626",
                    "Quest/Shell polybutylene pipe (1978–1995) degrades from "
                    "oxidants in municipal water. Class-action settlement (2001) "
                    "covers ~6M US homes. Catastrophic failure risk is 2–3× "
                    "standard copper — but only visible at interior inspection."))

    # ── CONSTRUCTION × PROTECTION CLASS ───────────────────────────────────────
    if frame and pc >= 8:
        out.append(("Frame Construction × PC ≥ 8",
                    m_overrides.get("frame_pc_high", 1.18), "#DC6803",
                    "ISO Protection Class 8–10 indicates fire station response "
                    "times >8 minutes. Frame construction reaches flashover in "
                    "4–6 minutes. The interaction creates a window where fire "
                    "response arrives after structural involvement — confirming "
                    "18% severity uplift across 75K+ fire incidents."))

    # ── PROTECTIVE MITIGANTS ───────────────────────────────────────────────────
    if metal and wf in ("Moderate", "High") and spk:
        out.append(("Metal Roof + Sprinkler × WF Zone (protective)",
                    m_overrides["metal_wf_sprinkler"], "#059669", ""))

    if ds >= 80 and wf in ("Moderate", "High"):
        out.append(("High Defensible Space (≥80) in WUI (protective)",
                    m_overrides["defensible_high"], "#059669",
                    "CA Regulation 2644.9 formally recognises defensible space as a "
                    "ratemaking factor. ≥100ft cleared zone reduces ember catch "
                    "probability by ~40%. This protective credit is the mitigation "
                    "story: same WUI zone, dramatically different risk profile."))

    if sup and new_r:
        out.append(("New Superior Construction ≤5yr (protective)",
                    m_overrides["new_superior"], "#059669", ""))

    # ── THREE-WAY SLOPE × WILDFIRE × POST-BURN RAIN ───────────────────────────
    if wf == "High" and slope > 55 and pbr > 60:
        out.append(("Slope >55% × High WF × Post-burn Rain >60 (3-way)",
                    m_overrides.get("slope_burn_rain", 1.45), "#DC2626",
                    "Post-fire debris flow cascade (Montecito 2018 archetype): Thomas Fire → "
                    "bare burned slopes + 0.5in/hr rain → catastrophic debris flow. USGS has "
                    "published 438+ post-fire debris flow assessments since 2013. CA CDI "
                    "formally recognises this fire-flood sequence. The three-way combination "
                    "compounds exponentially — no single-peril model captures this cascade."))
    elif wf == "High" and slope > 55:
        out.append(("Steep Slope >55% × High Wildfire Zone",
                    m_overrides.get("slope_burn_only", 1.18), "#DC6803",
                    "Steep terrain accelerates wildfire spread and increases post-fire "
                    "debris flow risk. Add post-burn rainfall intensity >60 to activate "
                    "the full three-way Montecito debris flow multiplier."))

    return out


def _el_to_score(el, arts):
    el_min = arts["el_min"]
    el_max = arts["el_max"]
    return float(np.clip(
        50 + 900 * (np.log1p(el) - np.log1p(el_min)) /
                   (np.log1p(el_max) - np.log1p(el_min)),
        50, 950
    ))


# ─────────────────────────────────────────────────────────────────────────────
# CORE INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _glm_baseline(inp: dict, arts: dict) -> tuple:
    """
    Compute regulatory GLM baseline for a single property.

    Uses PoissonRegressor (lambda_GLM) and GammaRegressor (mu_GLM) through the
    shared StandardScaler fitted on T12 features.  This is the PRICING CHAIN
    baseline — the authoritative Poisson x Gamma E[L].

    Returns: (lam_glm, mu_glm, glm_el, encoded_row)
      lam_glm  : annual claim frequency prediction from Poisson GLM
      mu_glm   : conditional severity prediction from Gamma GLM
      glm_el   : lam_glm x mu_glm  (E[L] baseline, M̂ = 1.0)
      row      : encoded DataFrame (reused for M̂ inference)
    """
    row    = _encode_row(inp)
    Xg     = row[arts["t12"]].values.astype(float)
    Xg_sc  = arts["glm_scaler"].transform(Xg)

    lam_glm = float(np.clip(arts["poisson_glm"].predict(Xg_sc)[0], 0.002, 0.30))
    mu_glm  = float(np.clip(arts["gamma_glm"].predict(Xg_sc)[0],   500.0, 600_000.0))
    glm_el  = lam_glm * mu_glm
    return lam_glm, mu_glm, glm_el, row


def _mhat_predict(row: pd.DataFrame, arts: dict) -> float:
    """
    Compute M̂ from the stacked ensemble (T3 features).
    M̂ = 1.0 means no interaction effect vs GLM baseline.
    M̂ > 1.0 = GLM under-prices this T3 co-exposure pattern.
    M̂ < 1.0 = GLM over-prices (protective interaction detected).
    """
    clip_lo = arts.get("oe_clip_lo", 0.10)
    clip_hi = arts.get("oe_clip_hi", 8.00)

    Xm = row[arts["t3"]].values
    meta_in = np.column_stack([
        arts["rf_m"].predict(Xm),
        arts["xgb_m"].predict(Xm),
        arts["lgb_m"].predict(Xm),
    ])
    m_hat = float(arts["iso_meta"].predict(
        [arts["ridge_meta"].predict(meta_in)[0]]
    )[0])
    return float(np.clip(m_hat, clip_lo, clip_hi))


def _freq_sev_display(inp: dict, cfg: dict, arts: dict) -> tuple:
    """
    ML HistGBM frequency + severity for Score A2 display and SHAP waterfall.
    DISPLAY ONLY — not in the pricing chain. Pricing uses _glm_baseline().
    """
    row      = _encode_row(inp)
    Xf       = row[arts["t12"]].values
    lam_disp = float(np.clip(arts["freq_model"].predict(Xf)[0], 0.005, cfg["lam_cap"]))
    Xs       = row[arts["sev_feats"]].values
    mu_disp  = float(np.clip(arts["sev_model"].predict(Xs)[0], 1_000, 500_000))
    return lam_disp, mu_disp, row


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE  λ_GLM × μ_GLM × 1.0  (pure GLM, no interaction multiplier)
# ─────────────────────────────────────────────────────────────────────────────
def predict_baseline(inp: dict, pricing_cfg: dict = None) -> dict:
    """
    Baseline score: Poisson GLM x Gamma GLM, M̂ = 1.0.

    This IS the industry gold-standard additive model — separate Poisson
    frequency and Gamma severity GLMs on T1+T2 features, no interaction terms.
    Used as the left panel in the M̂ Impact comparison tab.
    """
    cfg  = pricing_cfg if pricing_cfg is not None else DEFAULT_PRICING_CFG
    arts = load_arts()

    lam_glm, mu_glm, glm_el, _ = _glm_baseline(inp, arts)

    score   = _el_to_score(glm_el, arts)
    premium = glm_el / cfg["target_lr"]
    band, color  = _risk_band(score)
    action, acol = _uw_action(band)

    return dict(
        lambda_pred   = round(lam_glm, 5),
        mu_pred       = round(mu_glm, 2),
        m_hat         = 1.0,
        expected_loss = round(glm_el, 2),
        risk_score_a1 = round(score, 1),
        premium       = round(premium, 2),
        pure_premium  = round(glm_el / cfg["target_lr"], 2),
        risk_band     = band,
        risk_color    = color,
        uw_action     = action,
        uw_color      = acol,
        interactions  = [],
        warnings      = validate_inputs(inp),
    )


# Backward-compat alias
def predict_glm(inp: dict, pricing_cfg: dict = None) -> dict:
    return predict_baseline(inp, pricing_cfg)


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE  λ_GLM × μ_GLM × M̂  (Tier 3 interactions applied)
# ─────────────────────────────────────────────────────────────────────────────
def predict(inp: dict, pricing_cfg: dict = None) -> dict:
    """
    Full pipeline: (Poisson GLM x Gamma GLM) x M̂ → E[L] → risk score → premium.

    Architecture:
      1. _glm_baseline()    : E[L]_GLM = lambda_GLM x mu_GLM  (regulatory base)
      2. _mhat_predict()    : M̂ from stacked ensemble on T3 features
                              trained on GLM O/E residuals (Dai GLM Improver)
      3. E[L]_full = E[L]_GLM x M̂
      4. When M̂ = 1.0, collapses exactly to predict_baseline()

    No rule-based overrides are applied to M̂. The ensemble prediction is
    authoritative — protective and hazard interactions are learned from data.
    """
    cfg  = pricing_cfg if pricing_cfg is not None else DEFAULT_PRICING_CFG
    arts = load_arts()

    # ── Step 1: GLM baseline ──────────────────────────────────────────────────
    lam_glm, mu_glm, glm_el, row = _glm_baseline(inp, arts)

    # ── Step 2: M̂ ensemble (T3 features, GLM residual target) ────────────────
    m_hat = _mhat_predict(row, arts)

    # ── Step 3: Full E[L] = GLM_baseline × M̂ ─────────────────────────────────
    el      = glm_el * m_hat
    score   = _el_to_score(el, arts)
    premium = el / cfg["target_lr"]

    # ── Score A2: frequency + severity sub-scores (display, ML models) ────────
    lam_disp, mu_disp, _ = _freq_sev_display(inp, cfg, arts)
    lc = cfg["lam_cap"];  sc = cfg["sev_cap"]
    wf = cfg["score_wf"]; ws = cfg["score_ws"]
    f_score  = min(500.0, lam_disp / lc * 500)
    s_score  = min(500.0, (mu_disp * m_hat) / sc * 500)
    score_a2 = float(np.clip(
        (wf * (f_score + 1)**0.8 + ws * (s_score + 1)**0.8)**(1/0.8) - 1,
        0, 1000))

    band,  color  = _risk_band(score)
    action, acol  = _uw_action(band)
    interactions  = _get_interactions(inp, cfg.get("m_overrides"))

    # pct_from_tier3: portion of E[L] attributable to T3 interaction
    # Positive = M̂ adds risk above GLM baseline; negative = protective
    pct_from_t3 = (m_hat - 1.0) / max(m_hat, 0.01) * 100

    return dict(
        lambda_pred    = round(lam_glm, 5),    # Poisson GLM frequency
        mu_pred        = round(mu_glm, 2),     # Gamma GLM severity
        m_hat          = round(m_hat, 3),
        expected_loss  = round(el, 2),
        risk_score_a1  = round(score, 1),
        risk_score_a2  = round(score_a2, 1),
        f_score        = round(f_score, 1),
        s_score        = round(s_score, 1),
        premium        = round(premium, 2),
        pure_premium   = round(el / cfg["target_lr"], 2),
        pct_from_tier3 = round(pct_from_t3, 1),
        risk_band      = band,
        risk_color     = color,
        uw_action      = action,
        uw_color       = acol,
        interactions   = interactions,
        warnings       = validate_inputs(inp),
    )


def predict_both(inp: dict, pricing_cfg: dict = None) -> tuple:
    """
    Return (baseline_result, full_result).
    baseline = Poisson x Gamma GLM (M̂ = 1.0)
    full     = GLM x M̂ (Tier 3 interactions applied)
    The delta between the two IS the M̂ value proposition.
    """
    return predict_baseline(inp, pricing_cfg), predict(inp, pricing_cfg)


def predict_whatif(base_inp: dict, changes: dict, pricing_cfg: dict = None) -> dict:
    return predict({**base_inp, **changes}, pricing_cfg=pricing_cfg)


def compute_tier2_only_score(inp: dict, pricing_cfg: dict = None) -> dict:
    """Alias for predict_baseline() — M̂ forced to 1.0."""
    return predict_baseline(inp, pricing_cfg)


# ─────────────────────────────────────────────────────────────────────────────
# BATCH INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def batch_predict(df: pd.DataFrame,
                  pricing_cfg: dict = None,
                  sample_n: int = 5000) -> pd.DataFrame:
    """
    Batch inference for portfolio scatter / Tab 4 impact analysis.
    el_baseline = lambda_GLM x mu_GLM          (GLM additive, no M̂)
    el_full     = lambda_GLM x mu_GLM x M̂     (interaction-aware)

    Returns all columns app.py currently reads, including backward-compat aliases
    (el_glm = el_baseline, el_ml = el_full, etc.).
    """
    cfg  = pricing_cfg if pricing_cfg is not None else DEFAULT_PRICING_CFG
    arts = load_arts()

    samp = df.sample(min(sample_n, len(df)), random_state=42).copy()

    # Encode categoricals
    enc      = arts["encoders"]
    cat_cols = arts["cat_cols"]
    imp      = arts.get("impute_means", {})

    for col in cat_cols:
        if col not in samp.columns:
            continue
        le   = enc[col]
        raw  = samp[col].astype(str).values
        safe = np.where(np.isin(raw, le.classes_), raw, le.classes_[0])
        samp[col] = le.transform(safe)

    # Numeric imputation
    for col, fill in imp.items():
        if col in samp.columns:
            samp[col] = pd.to_numeric(samp[col], errors="coerce").fillna(fill)

    # Derived columns
    if "coverage_ratio" not in samp.columns and "coverage_amount" in samp.columns:
        hv = samp["home_value"].clip(lower=1)
        samp["coverage_ratio"] = (samp["coverage_amount"] / hv).clip(0.5, 2.0)

    if "credit_restricted" not in samp.columns and "state" in samp.columns:
        restricted_states = {"CA", "MA", "HI", "MD", "MI"}
        samp["credit_restricted"] = samp["state"].astype(str).isin(restricted_states).astype(int)

    # ── Poisson x Gamma GLM baseline ──────────────────────────────────────────
    Xg_sc   = arts["glm_scaler"].transform(samp[arts["t12"]].values.astype(float))
    lam_glm = np.clip(arts["poisson_glm"].predict(Xg_sc), 0.002, 0.30)
    mu_glm  = np.clip(arts["gamma_glm"].predict(Xg_sc),   500.0, 600_000.0)

    # ── M̂ ensemble (T3 features) ──────────────────────────────────────────────
    clip_lo = arts.get("oe_clip_lo", 0.10)
    clip_hi = arts.get("oe_clip_hi", 8.00)
    Xm      = samp[arts["t3"]].values
    meta_in = np.column_stack([
        arts["rf_m"].predict(Xm),
        arts["xgb_m"].predict(Xm),
        arts["lgb_m"].predict(Xm),
    ])
    m_hat = np.clip(
        arts["iso_meta"].predict(arts["ridge_meta"].predict(meta_in)),
        clip_lo, clip_hi
    )

    el_baseline = lam_glm * mu_glm          # GLM only, M̂ = 1.0
    el_full     = lam_glm * mu_glm * m_hat  # full pipeline

    el_min, el_max = arts["el_min"], arts["el_max"]
    def _score(el):
        return np.clip(
            50 + 900 * (np.log1p(el) - np.log1p(el_min)) /
                       (np.log1p(el_max) - np.log1p(el_min)),
            50, 950)

    # ML freq/sev for display columns (lambda_pred / mu_pred shown in scatter)
    Xf       = samp[arts["t12"]].values
    lam_disp = np.clip(arts["freq_model"].predict(Xf), 0.005, cfg["lam_cap"])
    Xs       = samp[arts["sev_feats"]].values
    mu_disp  = np.clip(arts["sev_model"].predict(Xs), 1_000, 500_000)

    return pd.DataFrame({
        "lambda_pred"        : lam_disp.round(5),      # ML display (Score A2)
        "mu_pred"            : mu_disp.round(0),        # ML display (Score A2)
        "m_hat_pred"         : m_hat.round(3),
        "el_baseline"        : el_baseline.round(2),   # GLM only (M̂=1.0)
        "el_full"            : el_full.round(2),        # GLM x M̂
        "score_baseline"     : _score(el_baseline).round(1),
        "score_full"         : _score(el_full).round(1),
        # True values from synthetic data (internal validation)
        "lambda_true"        : samp["lambda_true"].values,
        "mu_true"            : samp["mu_true"].values,
        "M_true"             : samp["M_true"].values,
        "expected_loss_true" : samp["expected_loss_true"].values,
        "claim_amount"       : samp["claim_amount"].values,
        "risk_score_true"    : samp["risk_score_true"].values,
        # Backward-compat aliases (app.py references these)
        "el_ml"              : el_full.round(2),
        "el_glm"             : el_baseline.round(2),
        "score_ml"           : _score(el_full).round(1),
        "score_glm"          : _score(el_baseline).round(1),
    })


# ─────────────────────────────────────────────────────────────────────────────
# GLM RELATIVITIES  (methodology tab — rate-filing display)
# ─────────────────────────────────────────────────────────────────────────────
def get_glm_relativities() -> dict:
    """
    Return rate relativities for the methodology tab.

    Returns a dict with three keys:
      "poisson"  : {feature -> exp(beta)} from PoissonRegressor (frequency GLM)
      "gamma"    : {feature -> exp(beta)} from GammaRegressor   (severity GLM)
      "tweedie"  : {feature -> exp(beta)} from TweedieRegressor (display artifact)

    The Poisson + Gamma relativities are the regulatory pricing artifacts.
    The Tweedie relativities are the traditional single-model view, retained
    for the rate-filing narrative section of the UI.

    Backward compat: also returns top-level Tweedie entries so code calling
    get_glm_relativities()["some_feature"] still works.
    """
    arts = load_arts()
    result = dict(
        poisson  = arts.get("poisson_relativities", {}),
        gamma    = arts.get("gamma_relativities",   {}),
        tweedie  = arts.get("glm_relativities",     {}),   # backward-compat label
    )
    # Also merge Tweedie entries at top level for backward-compat callers
    result.update(arts.get("glm_relativities", {}))
    return result


def get_shap_values(inp: dict) -> dict:
    """
    Compute SHAP decomposition for the three model components shown in Tab 3.

    Returns dict keyed by model display name (must match app.py tab iteration):
      "Frequency (λ)"  : ML HistGBM freq_model on T12 features  [display]
      "Severity (μ)"   : ML HistGBM sev_model  on T1+T2+T3 feats [display]
      "M-hat (M̂)"     : M̂ ensemble (xgb_m = HistGBM) on T3 features

    NOTE on architecture: Frequency and Severity SHAP are from the ML display
    models (not the Poisson/Gamma GLMs). They illustrate which T1+T2 features
    drive score A2 sub-components. The M-hat SHAP is the primary story — it
    shows which T3 co-exposure features drove M̂ away from 1.0 for this property,
    i.e. what the GLM residual pattern looks like for this risk profile.

    The key named "Poisson GLM (λ_GLM)" is returned as an additional entry
    for future app.py use (Tab 3 will be updated to show it). It uses
    shap.LinearExplainer for exact SHAP on the regulatory Poisson model.
    """
    import shap
    arts = load_arts()
    row  = _encode_row(inp)
    out  = {}

    # ── ML display models (TreeExplainer — fast, exact for tree models) ────────
    for name, model, feats in [
        ("Frequency (λ)",  arts["freq_model"], arts["t12"]),
        ("Severity (μ)",   arts["sev_model"],  arts["sev_feats"]),
        ("M-hat (M̂)",     arts["xgb_m"],      arts["t3"]),
    ]:
        try:
            expl = shap.TreeExplainer(model)
            sv   = expl.shap_values(row[feats].values)
            if isinstance(sv, list):
                sv = sv[1]
            base = expl.expected_value
            if isinstance(base, (list, np.ndarray)):
                base = float(base[0])
            out[name] = dict(values=sv[0].tolist(), features=list(feats), base=float(base))
        except Exception:
            out[name] = dict(values=[0.0]*len(feats), features=list(feats), base=0.0)

    # ── Poisson GLM SHAP via LinearExplainer (exact, O(M)) ────────────────────
    # Available for future app.py Tab 3 update — shows regulatory freq model drivers
    try:
        Xg_sc = arts["glm_scaler"].transform(row[arts["t12"]].values.astype(float))
        # Background = intercept (zero-mean scaled features → mean prediction)
        bg    = np.zeros((1, len(arts["t12"])))
        expl_p = shap.LinearExplainer(arts["poisson_glm"], bg, feature_dependence="independent")
        sv_p   = expl_p.shap_values(Xg_sc)
        base_p = float(arts["poisson_glm"].predict(bg)[0])
        out["Poisson GLM (λ_GLM)"] = dict(
            values   = sv_p[0].tolist(),
            features = list(arts["t12"]),
            base     = base_p,
        )
    except Exception:
        pass   # LinearExplainer may not be installed; non-fatal

    # ── Gamma GLM SHAP via LinearExplainer ────────────────────────────────────
    try:
        bg_g   = np.zeros((1, len(arts["t12"])))
        expl_g = shap.LinearExplainer(arts["gamma_glm"], bg_g, feature_dependence="independent")
        Xg_sc  = arts["glm_scaler"].transform(row[arts["t12"]].values.astype(float))
        sv_g   = expl_g.shap_values(Xg_sc)
        base_g = float(arts["gamma_glm"].predict(bg_g)[0])
        out["Gamma GLM (μ_GLM)"] = dict(
            values   = sv_g[0].tolist(),
            features = list(arts["t12"]),
            base     = base_g,
        )
    except Exception:
        pass

    return out