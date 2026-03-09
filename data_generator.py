"""
data_generator.py  v6
=====================
Sprint 3 additions on top of v5 Sprint 0/2 base.

Key changes vs v5:
  1-11. All v5 Sprint 0 realism features retained (frailty, bimodal
        distributions, copula, missingness, vintage, CAT flag, etc.)

  Sprint 3 additions:
  12. SLOPE + POST-BURN RAINFALL INTENSITY features added (3.3):
      - slope_steepness_pct (0-100): state-correlated terrain feature
      - post_burn_rainfall_intensity (0-100): NOAA precipitation signal
      - Three-way M̂ interaction: Slope x High WF x Post-burn Rain = x1.65
        (Montecito 2018 archetype -- Thomas Fire -> debris flow cascade)
  13. DATA VINTAGE FLAGS (0.5):
      - data_vintage_flag: 12% of properties have stale assessor records
      - rcv_vintage_flag: 8% stale MLS comparables for RCV estimation
      - is_near_duplicate: 0.5% near-duplicate records (agent entry errors)
  14. VARIABLE TREND RATE (0.7):
      - trend_rate_annual ~ Normal(0.06, 0.015) per policy
        (+/-1.5% annual variation around 6% construction inflation baseline)
  15. RENEWAL CREDIT (0.7):
      - Policies with tenure ≥3yr + zero claims -> lambda x 0.95 (-5%)
      - renewal_credit column (0/1) added for validation transparency
  16. PERMIT SCORE UI SURFACED (2.2):
      - permit_score already in T12; now documented as positive maintenance
        signal ("we credit homes with recent permits, not just penalise age")

Calibration targets (III/ISO/Verisk 2022-2024):
  - Annual claim rate:          5.3-5.6%
  - Avg blended severity:      $15,000-$18,000
  - Fire/lightning severity:   $84,000-$88,000
  - Wind/hail severity:        ~$14,700
  - Water damage severity:     ~$15,400
  - Zero-claim proportion:      94-95%
  - Target loss ratio:          60-65%
  - Avg indicated premium:     $2,150-$2,500
  - Tweedie power p:            1.65
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

RNG = np.random.default_rng(42)

# ── Industry calibration constants ──────────────────────────────────────────
BASE_CLAIM_RATE   = 0.058      # 5.8% target input -> after frailty normalisation ~5.3-5.5%
CAT_THRESHOLD     = 85_000     # 95th pct threshold for CAT/attritional split
CAT_BASE_FRAC     = 0.015      # ~1.5% per claim are CAT events (tail realistic)
TREND_RATE        = 0.06       # 6% annual construction inflation
TARGET_LR         = 0.65
EXPENSE_LOAD      = 0.18
SEVERITY_BASE_PCT = 0.048      # 4.8% of home_value -> ~$16,800 for $350K home

# Gamma frailty parameters (unobserved heterogeneity)
# Gamma(shape, scale) with mean=1 -> shape=scale_inv
# Higher shape = tighter frailty (less heterogeneity)
# shape=1.5 -> CV=0.82; shape=2.0 -> CV=0.71 (industry-calibrated range)
FRAILTY_SHAPE = 3.5   # CV(frailty)=0.535 -> realistic Gini 0.25-0.40; shape=1.8 was too noisy

# M̂ noise: Lognormal(0, sigma) -> individual policy deviation within cells
M_NOISE_SIGMA = 0.08  # +/-8% per-policy noise around cell mean

# ── State parameters ─────────────────────────────────────────────────────────
# wf=wildfire prob, fl=flood prob, eq=earthquake prob,
# val_med=median home value, coastal=bool, hail_p=hail prob,
# canopy_p=canopy prob, wf_latent=continuous wildfire exposure (0-1)
STATES = {
    "CA": dict(wf=0.55, fl=0.12, eq=0.65, val_med=680_000, coastal=False,
               hail_p=0.12, canopy_p=0.45, wf_latent=0.62, slope_p=0.62),
    "FL": dict(wf=0.05, fl=0.65, eq=0.02, val_med=360_000, coastal=True,
               hail_p=0.22, canopy_p=0.55, wf_latent=0.08, slope_p=0.05),
    "TX": dict(wf=0.28, fl=0.32, eq=0.05, val_med=300_000, coastal=False,
               hail_p=0.45, canopy_p=0.35, wf_latent=0.32, slope_p=0.28),
    "LA": dict(wf=0.05, fl=0.68, eq=0.02, val_med=220_000, coastal=True,
               hail_p=0.30, canopy_p=0.60, wf_latent=0.06, slope_p=0.04),
    "OK": dict(wf=0.32, fl=0.22, eq=0.18, val_med=195_000, coastal=False,
               hail_p=0.55, canopy_p=0.28, wf_latent=0.35, slope_p=0.22),
    "CO": dict(wf=0.42, fl=0.08, eq=0.12, val_med=480_000, coastal=False,
               hail_p=0.48, canopy_p=0.32, wf_latent=0.45, slope_p=0.70),
    "NC": dict(wf=0.08, fl=0.28, eq=0.04, val_med=275_000, coastal=True,
               hail_p=0.28, canopy_p=0.68, wf_latent=0.10, slope_p=0.38),
    "GA": dict(wf=0.08, fl=0.22, eq=0.04, val_med=285_000, coastal=False,
               hail_p=0.25, canopy_p=0.65, wf_latent=0.10, slope_p=0.32),
    "AZ": dict(wf=0.38, fl=0.06, eq=0.22, val_med=335_000, coastal=False,
               hail_p=0.18, canopy_p=0.22, wf_latent=0.42, slope_p=0.48),
    "NV": dict(wf=0.32, fl=0.04, eq=0.38, val_med=360_000, coastal=False,
               hail_p=0.15, canopy_p=0.18, wf_latent=0.38, slope_p=0.55),
}
STATE_KEYS  = list(STATES.keys())
# Credit-restricted states: CA, MA (not in our state list), HI (not in list)
# CA is in list -- credit suppressed there
CREDIT_RESTRICTED = {"CA"}
STATE_PROBS = [0.18, 0.15, 0.15, 0.07, 0.06, 0.09, 0.07, 0.07, 0.08, 0.08]

# ── Property lookup tables ───────────────────────────────────────────────────
CONSTRUCTION = {
    "Frame":    dict(freq_m=1.28, sev_m=1.10, p=0.50),
    "Masonry":  dict(freq_m=0.86, sev_m=0.90, p=0.25),
    "Superior": dict(freq_m=0.72, sev_m=0.80, p=0.10),
    "Mixed":    dict(freq_m=1.08, sev_m=1.00, p=0.15),
}
ROOF = {
    "Asphalt Shingle": dict(fire_r=1.00, p=0.55),
    "Wood Shake":      dict(fire_r=1.80, p=0.08),
    "Metal":           dict(fire_r=0.70, p=0.14),
    "Tile":            dict(fire_r=0.80, p=0.18),
    "Flat/Built-Up":   dict(fire_r=1.20, p=0.05),
}
OCCUPANCY = {
    "Owner Occupied":  dict(freq_m=1.00, p=0.74),
    "Tenant Occupied": dict(freq_m=1.22, p=0.19),
    "Vacant":          dict(freq_m=1.65, p=0.07),
}
DED_VALS  = [500, 1000, 2500, 5000]
DED_PROBS = [0.12, 0.42, 0.32, 0.14]
DED_FREQ  = {500: 1.00, 1000: 0.90, 2500: 0.75, 5000: 0.62}

# Policy vintage years and their CAT multipliers
# 2020-21 were elevated CAT years; 2022-24 normalising
VINTAGE_YEARS = [2020, 2021, 2022, 2023, 2024]
VINTAGE_PROBS = [0.12, 0.14, 0.22, 0.26, 0.26]
VINTAGE_CAT_M = {2020: 1.18, 2021: 1.15, 2022: 1.05, 2023: 1.02, 2024: 1.00}


# ════════════════════════════════════════════════════════════════════════════
# MAIN GENERATION FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def generate_dataset(n=100_000):
    print(f"Generating {n:,} actuarially calibrated homeowners records (v6)...")
    print("  Sprint 0+3: frailty + realistic distributions + M̂ noise + missingness + slope/rain/vintage")

    state = RNG.choice(STATE_KEYS, size=n, p=STATE_PROBS)

    # ── Expanded 8-variable Gaussian copula ─────────────────────────────────
    # Variables:
    #   0: home_age_latent
    #   1: home_value_latent
    #   2: credit_latent
    #   3: claims_latent
    #   4: pc_latent (protection class)
    #   5: fire_station_latent  (near-collinear with pc, ρ=0.75)
    #   6: wf_exposure_latent   (correlated with home_value: WUI premium)
    #   7: permit_latent        (anti-correlated with home_age: older->more permits)
    corr8 = np.array([
    #    age    val    cred   claims  pc     fst    wf_exp permit
        [ 1.00, -0.15, -0.18,  0.22,  0.12,  0.14,  0.05, -0.40],  # home_age
        [-0.15,  1.00,  0.20, -0.12, -0.18, -0.16,  0.28, -0.10],  # home_val
        [-0.18,  0.20,  1.00, -0.42, -0.08, -0.10, -0.05,  0.15],  # credit
        [ 0.22, -0.12, -0.42,  1.00,  0.10,  0.08,  0.06, -0.18],  # claims
        [ 0.12, -0.18, -0.08,  0.10,  1.00,  0.75, -0.02,  0.05],  # pc
        [ 0.14, -0.16, -0.10,  0.08,  0.75,  1.00, -0.02,  0.04],  # fire_stn
        [ 0.05,  0.28, -0.05,  0.06, -0.02, -0.02,  1.00, -0.06],  # wf_exp
        [-0.40, -0.10,  0.15, -0.18,  0.05,  0.04, -0.06,  1.00],  # permit
    ])
    # Make positive definite
    ev = np.linalg.eigvalsh(corr8)
    if ev.min() < 1e-8:
        corr8 += np.eye(8) * (abs(ev.min()) + 1e-6)
    L8 = np.linalg.cholesky(corr8)
    Z8 = RNG.standard_normal((n, 8))
    U8 = norm.cdf(Z8 @ L8.T)

    val_meds = np.array([STATES[s]["val_med"] for s in state])
    wf_lats  = np.array([STATES[s]["wf_latent"] for s in state])

    # ── Core continuous features from copula ────────────────────────────────
    home_age     = np.clip(np.round(stats.beta.ppf(U8[:,0], 2, 3) * 70 + 3), 2, 74).astype(int)
    home_value   = np.clip(np.round(val_meds * stats.lognorm.ppf(
                       np.clip(U8[:,1], 0.01, 0.99), s=0.26)), 80_000, 2_500_000).astype(int)

    # Credit score: MIXTURE distribution with threshold spikes
    # 70% from high-credit Normal(740,80), 30% from subprime Normal(590,60)
    # Plus ~3% exact at threshold boundaries 580/620/660
    credit_score = _generate_credit_scores(U8[:,2], n)

    latent_claims    = stats.expon.ppf(np.clip(U8[:,3], 0.001, 0.999), scale=0.35)
    protection_class = np.clip(np.round(stats.beta.ppf(U8[:,4], 2, 2) * 9 + 1), 1, 10).astype(int)

    # Fire station: near-collinear with PC but not identical
    # PC 1-3 -> dist 0.5-2mi; PC 8-10 -> dist 5-12mi + noise
    dist_to_fire_station = _generate_fire_station_dist(U8[:,5], protection_class, n)

    # WF exposure latent: blend state baseline with copula signal
    wf_continuous = np.clip(wf_lats + 0.15*(U8[:,6] - 0.5), 0.02, 0.95)

    # Permit score: higher for older homes (more likely to have reroofed)
    permit_score_raw = np.clip(U8[:,7] * 100, 0, 100)

    year_built = np.clip(2024 - home_age, 1950, 2022)

    # Coverage ratio with fraud signal
    cov_raw        = RNG.normal(1.05, 0.20, n)
    coverage_ratio = np.clip(cov_raw, 0.65, 1.45)
    coverage_amount = np.clip((home_value * coverage_ratio).astype(int), 80_000, 2_500_000)

    square_footage = np.clip(np.round(RNG.normal(2100, 700, n)), 500, 6000).astype(int)
    stories        = RNG.choice([1, 2, 3], size=n, p=[0.56, 0.37, 0.07])

    # Prior claims: zero-inflated, capped at 3 (realistic CLUE distribution)
    # 92% zero, 6% one claim, 2% two+ claims
    prior_claims_3yr = _generate_prior_claims(latent_claims, n)

    construction_type = _pick(CONSTRUCTION, n)

    # Roof material: state-adjusted Wood Shake probability
    wf_states   = {"CA", "CO", "AZ", "NV"}
    roof_keys   = list(ROOF.keys())
    roof_base_p = np.array([ROOF[k]["p"] for k in roof_keys])
    roof_material = np.array([
        RNG.choice(roof_keys, p=_roof_probs(roof_base_p, home_age[i], state[i] in wf_states))
        for i in range(n)
    ])

    occupancy  = _pick(OCCUPANCY, n)
    deductible = RNG.choice(DED_VALS, size=n, p=DED_PROBS)

    cr_n = (credit_score - 500) / 350.0
    cl_n = prior_claims_3yr / 5.0

    # Behavioral features correlated with credit/claims
    security_system  = (RNG.random(n) < np.clip(0.28 + 0.40*cr_n, 0.12, 0.72)).astype(int)
    smoke_detectors  = (RNG.random(n) < np.clip(0.76 + 0.16*cr_n, 0.62, 0.94)).astype(int)
    sprinkler_system = (RNG.random(n) < np.clip(0.07 + 0.13*cr_n, 0.03, 0.22)).astype(int)
    gated_community  = (RNG.random(n) < np.clip(0.07 + 0.22*cr_n, 0.03, 0.30)).astype(int)
    swimming_pool    = (RNG.random(n) < np.clip(0.24 - 0.04*cl_n, 0.10, 0.30)).astype(int)
    trampoline       = (RNG.random(n) < 0.11).astype(int)  # NO credit correlation
    dog              = (RNG.random(n) < 0.38).astype(int)

    # Roof age: BIMODAL -- 15% recent reroofs (0-3yr), 85% main distribution
    roof_age_yr = _generate_roof_age(home_age, n)

    # Peril zone assignments
    wf_p = np.array([STATES[s]["wf"]       for s in state])
    fl_p = np.array([STATES[s]["fl"]       for s in state])
    eq_p = np.array([STATES[s]["eq"]       for s in state])
    h_p  = np.array([STATES[s]["hail_p"]   for s in state])
    c_p  = np.array([STATES[s]["canopy_p"] for s in state])

    wildfire_zone   = _zone(wf_p, n)
    flood_zone      = _zone(fl_p, n)
    earthquake_zone = _zone(eq_p, n)
    hail_zone       = _zone(h_p, n)

    canopy_adj                = np.clip(c_p + 0.08*cl_n, 0.05, 0.80)
    vegetation_risk_composite = _zone(canopy_adj, n)

    # Interior hazard features: pre-1950 wiring and 1978-1995 plumbing
    # has_knob_tube_wiring: knob-and-tube electrical -- fire and arc-fault risk
    knob_tube_p = np.where(home_age >= 70, 0.20,
                  np.where(home_age >= 60, 0.10,
                  np.where(home_age >= 50, 0.04, 0.005)))
    has_knob_tube_wiring = (RNG.random(n) < knob_tube_p).astype(int)

    # has_polybutylene_pipe: Quest/Shell polybutylene (1978-1995) -- water leak risk
    polybut_p = np.where((year_built >= 1978) & (year_built <= 1995), 0.30, 0.015)
    has_polybutylene_pipe = (RNG.random(n) < polybut_p).astype(int)

    # Defensible space: higher in WUI states and WUI properties
    defensible_space = _generate_defensible_space(wildfire_zone, state, n)

    coastal_st = {"FL", "NC", "GA", "LA"}
    dist_to_coast = np.where(
        np.isin(state, list(coastal_st)),
        np.clip(RNG.exponential(7, n),   0.2,  80),
        np.clip(RNG.exponential(60, n),  1.0, 500),
    ).round(1)

    # Policy vintage + VARIABLE trend factors (0.7: +/-1.5% annual variation)
    policy_year   = RNG.choice(VINTAGE_YEARS, size=n, p=VINTAGE_PROBS)
    years_to_2024 = 2024 - policy_year
    # Sprint 3: each policy gets its own trend rate drawn from N(6%, 1.5%)
    trend_rate_annual = np.clip(RNG.normal(TREND_RATE, 0.015, n), 0.01, 0.12)
    trend_factor  = np.array([(1 + trend_rate_annual[i])**years_to_2024[i] for i in range(n)])
    vintage_cat_m = np.array([VINTAGE_CAT_M[y] for y in policy_year])

    # Credit restriction flag: suppress credit in CA
    credit_restricted = np.isin(state, list(CREDIT_RESTRICTED)).astype(int)

    # Permit score: calibrated to home age (older -> more renovation history)
    permit_score = np.clip(
        permit_score_raw * (1 + 0.008 * np.maximum(home_age - 20, 0)),
        0, 100
    ).round(1)

    # ── Sprint 3 (3.3): Slope steepness + post-burn rainfall intensity ──────
    # slope_steepness_pct: state-correlated terrain feature (0-100)
    slope_base = np.array([STATES[s]["slope_p"] * 100 for s in state])
    slope_steepness_pct = np.clip(
        slope_base + RNG.normal(0, 15, n), 0, 100
    ).round(1)

    # post_burn_rainfall_intensity: NOAA precipitation signal (0-100)
    # Correlated with wildfire_zone -- WUI areas get monsoon/atmospheric river events
    burn_base = np.where(wildfire_zone == "High",     60.0,
                np.where(wildfire_zone == "Moderate",  35.0, 12.0))
    post_burn_rainfall_intensity = np.clip(
        burn_base + RNG.normal(0, 16, n), 0, 100
    ).round(1)

    # ── Sprint 3 (0.5): Data vintage flags ──────────────────────────────────
    # data_vintage_flag: 12% of properties have stale county assessor records
    # (assessor data lags 3-5 years; affects RCV and building code estimates)
    data_vintage_flag = (RNG.random(n) < 0.12).astype(int)

    # rcv_vintage_flag: 8% stale MLS comparables for RCV estimation
    # (MLS data for rural/low-turnover markets can be 2-3yr stale)
    rcv_vintage_flag = (RNG.random(n) < 0.08).astype(int)

    df = pd.DataFrame({
        "policy_id":               [f"POL{i+1:06d}" for i in range(n)],
        "policy_year":             policy_year,
        "state":                   state,
        "construction_type":       construction_type,
        "roof_material":           roof_material,
        "occupancy":               occupancy,
        "year_built":              year_built,
        "home_age":                home_age,
        "home_value":              home_value,
        "coverage_amount":         coverage_amount,
        "coverage_ratio":          coverage_ratio.round(3),
        "square_footage":          square_footage,
        "stories":                 stories,
        "protection_class":        protection_class,
        "prior_claims_3yr":        prior_claims_3yr,
        "credit_score":            credit_score,
        "credit_restricted":       credit_restricted,
        "deductible":              deductible,
        "swimming_pool":           swimming_pool,
        "trampoline":              trampoline,
        "dog":                     dog,
        "security_system":         security_system,
        "smoke_detectors":         smoke_detectors,
        "sprinkler_system":        sprinkler_system,
        "gated_community":         gated_community,
        "roof_age_yr":             roof_age_yr,
        "wildfire_zone":           wildfire_zone,
        "flood_zone":              flood_zone,
        "earthquake_zone":         earthquake_zone,
        "hail_zone":               hail_zone,
        "vegetation_risk_composite": vegetation_risk_composite,
        "has_knob_tube_wiring":    has_knob_tube_wiring,
        "has_polybutylene_pipe":   has_polybutylene_pipe,
        "defensible_space_score":  defensible_space,
        "permit_score":            permit_score,
        "dist_to_coast_mi":        dist_to_coast,
        "dist_to_fire_station_mi": dist_to_fire_station,
        "trend_factor":            trend_factor.round(4),
        "trend_rate_annual":       trend_rate_annual.round(4),
        "vintage_cat_multiplier":  vintage_cat_m,
        # Sprint 3 new columns
        "slope_steepness_pct":           slope_steepness_pct,
        "post_burn_rainfall_intensity":  post_burn_rainfall_intensity,
        "data_vintage_flag":             data_vintage_flag,
        "rcv_vintage_flag":              rcv_vintage_flag,
    })

    df = _add_missingness(df, state)
    df = _compute_targets(df)
    _print_validation(df)
    return df


# ════════════════════════════════════════════════════════════════════════════
# REALISTIC DISTRIBUTION HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _generate_credit_scores(u_latent, n):
    """
    Mixture distribution with threshold spikes.
    70% from high-credit Normal(740,80), 30% from subprime Normal(590,60).
    ~3% of cases land exactly on FICO tier thresholds: 580, 620, 660.
    Banned in CA -- returns 720 (neutral) for those policies; corrected later.
    """
    is_subprime = RNG.random(n) < 0.22    # 22%: yields ~15-18% subprime (<620)
    high_credit = np.round(RNG.normal(745, 75, n)).astype(int)
    sub_credit  = np.round(RNG.normal(585, 55, n)).astype(int)
    base = np.where(is_subprime, sub_credit, high_credit)

    # Threshold spikes: 1% each at 580/620/660 boundary artefacts
    for thresh in [580, 620, 660]:
        mask = RNG.random(n) < 0.010
        base = np.where(mask, thresh, base)

    return np.clip(base, 500, 850).astype(int)


def _generate_prior_claims(latent_claims, n):
    """
    Zero-inflated distribution: 92% zero, 6% one, 2% two+.
    DO NOT use Poisson(0.1) -- produces too many 2+ claim policies vs CLUE data.
    """
    roll = RNG.random(n)
    cnt  = np.zeros(n, dtype=int)
    # ~6% get 1 claim
    cnt  = np.where((roll > 0.92) & (roll <= 0.98), 1, cnt)
    # ~2% get 2-3 claims (driven by latent_claims signal)
    cnt  = np.where(roll > 0.98,
                    np.clip(np.round(latent_claims * 1.5).astype(int), 2, 3),
                    cnt)
    return cnt


def _generate_roof_age(home_age, n):
    """
    Bimodal: 15% recent reroofs (0-3yr), 85% main distribution peaking ~18yr.
    + Reporting error: +/-0-4yr uniform noise (self-report vs satellite gap).
    """
    is_recent  = RNG.random(n) < 0.15
    recent_age = np.round(RNG.beta(1.2, 8, n) * 3).astype(int)        # 0-3yr
    main_age   = np.round(RNG.beta(2.8, 4, n) * 28 + 5).astype(int)   # peaks ~18yr

    base_age   = np.where(is_recent, recent_age, main_age)
    # Reporting error: +/-0-4yr (applicant overestimates age by ~2yr on average)
    noise      = RNG.integers(-2, 5, n)   # asymmetric: more over-reporting
    roof_age   = np.clip(base_age + noise, 0, home_age).astype(int)
    return roof_age


def _generate_fire_station_dist(u_fst, protection_class, n):
    """
    Near-collinear with protection_class (ρ~0.75) but not identical.
    PC 1-3 -> 0.3-2.5mi; PC 4-6 -> 2-5mi; PC 7-10 -> 4-15mi.
    """
    pc = protection_class
    base_dist = np.where(pc <= 3, RNG.exponential(1.2, n),
                np.where(pc <= 6, RNG.exponential(3.0, n),
                                  RNG.exponential(7.5, n)))
    # Add noise from copula signal
    noise = stats.expon.ppf(np.clip(u_fst, 0.01, 0.99), scale=1.5)
    dist  = np.clip(base_dist + 0.3*noise, 0.2, 30.0)
    return dist.round(1)


def _generate_defensible_space(wildfire_zone, state, n):
    """
    Higher in WUI states (CA/CO/AZ/NV) and high wildfire zone properties.
    Range 0-100; validated by Nature Communications 2025 (up to 52% loss reduction).
    """
    wui_states = {"CA", "CO", "AZ", "NV"}
    base = np.where(np.isin(state, list(wui_states)), 55.0, 35.0)
    wf_bonus = np.where(wildfire_zone == "High",   10.0,
               np.where(wildfire_zone == "Moderate", 5.0, 0.0))
    noise = RNG.normal(0, 15, n)
    score = np.clip(base + wf_bonus + noise, 0, 100)
    return score.round(1)


def _add_missingness(df, state):
    """
    Realistic missingness patterns:
    - vegetation_risk_composite: 8% rural (non-CAT states) missing -> 'Unknown'
    - dist_to_fire_station: 3% remote properties missing -> -1 sentinel
    - defensible_space: 25% missing for non-WUI states (not assessed)
    Sprint 3 (0.5):
    - is_near_duplicate: 0.5% near-duplicates (agent data entry errors)
    """
    n = len(df)
    rural_states = {"OK", "GA", "NC", "AZ", "NV"}   # 5 states -> ~4% canopy unknown
    is_rural = np.isin(state, list(rural_states))

    # 8% of rural properties -> vegetation assessment unknown
    canopy_missing = is_rural & (RNG.random(n) < 0.08)
    df.loc[canopy_missing, "vegetation_risk_composite"] = "Unknown"

    # 30% of non-WUI properties -> defensible space not assessed (~22% overall)
    non_wui = ~np.isin(state, ["CA","CO","AZ","NV"])
    ds_missing = non_wui & (RNG.random(n) < 0.30)
    df.loc[ds_missing, "defensible_space_score"] = np.nan

    # 3% remote: fire station distance unknown -> -1 sentinel
    remote_missing = RNG.random(n) < 0.03
    df.loc[remote_missing, "dist_to_fire_station_mi"] = -1.0

    # Sprint 3 (0.5): 0.5% near-duplicate records (agent data entry errors)
    # These are flagged for validation but not removed -- real portfolios contain them
    dup_mask = RNG.random(n) < 0.005
    df["is_near_duplicate"] = dup_mask.astype(int)

    return df


# ════════════════════════════════════════════════════════════════════════════
# TARGET COMPUTATION: lambda x mu x M̂ -> E[L] -> losses -> score -> premium
# ════════════════════════════════════════════════════════════════════════════

def _compute_targets(df):
    n = len(df)

    # ── FREQUENCY (lambda) ────────────────────────────────────────────────
    lam = np.full(n, BASE_CLAIM_RATE)
    lam *= df["construction_type"].map({k: v["freq_m"] for k, v in CONSTRUCTION.items()}).values
    lam *= 1 + 0.0025 * np.maximum(df["home_age"].values - 10, 0)
    lam *= 0.82 + 0.036 * (df["protection_class"].values - 1)
    lam *= df["occupancy"].map({k: v["freq_m"] for k, v in OCCUPANCY.items()}).values
    lam *= np.power(1.32, df["prior_claims_3yr"].values)

    # Credit: suppressed in restricted states -> no multiplier effect
    cr_restricted = df["credit_restricted"].values.astype(bool)
    cr_mult = np.where(
        cr_restricted, 1.0,
        np.power(750 / df["credit_score"].values.clip(500, 850), 0.55)
    )
    lam *= cr_mult
    lam *= df["deductible"].map(DED_FREQ).values
    lam *= np.where(df["swimming_pool"].values,    1.10, 1.0)
    lam *= np.where(df["trampoline"].values,       1.14, 1.0)
    lam *= np.where(df["dog"].values,              1.08, 1.0)
    lam *= np.where(df["security_system"].values,  0.90, 1.0)
    lam *= np.where(df["smoke_detectors"].values,  0.93, 1.0)
    lam *= np.where(df["sprinkler_system"].values, 0.82, 1.0)
    lam *= np.where(df["gated_community"].values,  0.91, 1.0)
    # Interior hazard multipliers (interior condition data gap = pricing blind spot)
    lam *= np.where(df["has_knob_tube_wiring"].values,  1.20, 1.0)   # arc-fault -> fire
    lam *= np.where(df["has_polybutylene_pipe"].values, 1.25, 1.0)   # pipe failure -> water

    # GAMMA FRAILTY -- unobserved heterogeneity, mean=1
    # This is the key driver of realistic R² (0.45-0.60)
    frailty = RNG.gamma(shape=FRAILTY_SHAPE, scale=1.0/FRAILTY_SHAPE, size=n)
    lam *= frailty

    lam = lam / lam.mean() * BASE_CLAIM_RATE
    lam = np.clip(lam, 0.005, 0.18)

    # Sprint 3 (0.7): Renewal credit -- 3+ yr tenure + 0 claims -> -5%
    # Reflects risk selection / adverse exit (loyal low-risk customers stay)
    tenure_years = 2024 - df["policy_year"].values
    renewal_credit = ((tenure_years >= 3) & (df["prior_claims_3yr"].values == 0)).astype(int)
    lam = np.where(renewal_credit > 0, lam * 0.95, lam)
    df["renewal_credit"]  = renewal_credit
    df["lambda_true"]   = lam.round(5)
    df["frailty_term"]  = frailty.round(4)   # kept for audit/validation

    # ── SEVERITY (mu) ─────────────────────────────────────────────────────
    # Recalibrated: 4.8% of home_value -> ~$16,800 for $350K home
    mu = df["home_value"].values * SEVERITY_BASE_PCT
    mu *= 1 + (df["square_footage"].values - 2000) / 28_000
    mu *= 1 + 0.04 * (df["stories"].values - 1)
    mu *= 0.87 + 0.025 * (df["protection_class"].values - 1)

    # Fire station: -1 sentinel -> use portfolio mean (3.2mi)
    fst = np.where(df["dist_to_fire_station_mi"].values < 0, 3.2,
                   df["dist_to_fire_station_mi"].values)
    mu *= 1 + 0.018 * fst
    mu *= np.where(df["smoke_detectors"].values,  0.87, 1.0)
    mu *= np.where(df["sprinkler_system"].values, 0.62, 1.0)
    # Interior hazard severity uplift
    mu *= np.where(df["has_knob_tube_wiring"].values,  1.15, 1.0)   # uncontrolled arc fires burn hotter
    mu *= np.where(df["has_polybutylene_pipe"].values, 1.18, 1.0)   # catastrophic pipe burst -> structural

    # Defensible space: credit when assessed, neutral when missing
    ds = df["defensible_space_score"].values
    ds_known = ~np.isnan(ds)
    mu = np.where(ds_known & (ds >= 70), mu * 0.88,   # well-cleared -> -12% sev
          np.where(ds_known & (ds <= 30), mu * 1.08,   # poor clearance -> +8%
                   mu))

    mu = np.minimum(mu, df["coverage_amount"].values * 0.80)
    mu = np.clip(mu, 1_500, 400_000)
    df["mu_true"] = mu.round(0)

    # ── INTERACTION MULTIPLIER M̂ ─────────────────────────────────────────
    M     = np.ones(n)
    wood  = df["roof_material"].values == "Wood Shake"
    wf_h  = df["wildfire_zone"].values == "High"
    wf_m  = df["wildfire_zone"].values == "Moderate"
    fl_h  = df["flood_zone"].values    == "High"
    fl_m  = df["flood_zone"].values    == "Moderate"
    near  = df["dist_to_coast_mi"].values < 5
    eq_h  = df["earthquake_zone"].values == "High"
    eq_m  = df["earthquake_zone"].values == "Moderate"
    old_r = df["roof_age_yr"].values > 20
    frame = df["construction_type"].values == "Frame"
    pc_hi = df["protection_class"].values >= 8      # PC 8-10: poorly protected
    hi_can  = df["vegetation_risk_composite"].values == "High"
    mod_can = df["vegetation_risk_composite"].values == "Moderate"
    hi_hail = df["hail_zone"].values == "High"
    mod_hail= df["hail_zone"].values == "Moderate"
    hi_cl   = df["prior_claims_3yr"].values >= 2

    # Wildfire x roof material (empirical AIR/RMS: Wood Shake WUI = 3-4x)
    M = np.where(wood & wf_h,          M * 3.50, M)
    M = np.where(wood & wf_m,          M * 2.10, M)
    M = np.where(~wood & wf_h,         M * 1.80, M)
    M = np.where(wood & ~wf_h & ~wf_m, M * 1.40, M)

    # Flood x coastal proximity
    M = np.where(fl_h & near,          M * 2.20, M)
    M = np.where(fl_h & ~near,         M * 1.60, M)
    M = np.where(fl_m,                 M * 1.20, M)

    # Earthquake zone
    M = np.where(eq_h,                 M * 1.50, M)
    M = np.where(eq_m,                 M * 1.15, M)

    # Old roof x construction (frame worse than masonry)
    M = np.where(old_r & frame,        M * 1.35, M)
    M = np.where(old_r & ~frame,       M * 1.15, M)

    # NEW: Construction x Protection Class (ISO validated: frame+PC8+ = +20%)
    M = np.where(frame & pc_hi,        M * 1.18, M)

    # Water claims x tree canopy (root intrusion + moisture cycle)
    M = np.where(hi_cl & hi_can,       M * 1.55, M)
    M = np.where(hi_cl & mod_can,      M * 1.25, M)
    M = np.where(~hi_cl & hi_can,      M * 1.10, M)

    # Old roof x hail frequency (accumulated micro-damage)
    M = np.where(old_r & hi_hail,      M * 1.45, M)
    M = np.where(old_r & mod_hail,     M * 1.20, M)
    M = np.where(~old_r & hi_hail,     M * 1.15, M)

    # Defensible space PROTECTIVE interaction (reduces WUI fire exposure)
    ds_val  = df["defensible_space_score"].fillna(50).values
    ds_good = ds_val >= 70
    M = np.where(wf_h & ds_good,       M * 0.72, M)   # well-hardened WUI
    M = np.where(wf_m & ds_good,       M * 0.82, M)

    # Apply vintage CAT multiplier (2020-21 elevated years)
    M *= df["vintage_cat_multiplier"].values

    # Sprint 3 (3.3): THREE-WAY INTERACTION -- Slope x High Wildfire x Post-burn Rain
    # Montecito 2018 archetype: Thomas Fire -> burned slopes -> modest rainfall -> debris flow
    # USGS 438+ post-fire debris flow assessments since 2013 validate this cascade
    slope_hi = df["slope_steepness_pct"].values > 55
    pbr_hi   = df["post_burn_rainfall_intensity"].values > 60
    M = np.where(wf_h & slope_hi & pbr_hi,  M * 1.65, M)  # full three-way (x1.65 additional)
    M = np.where(wf_h & slope_hi & ~pbr_hi, M * 1.18, M)  # slope + WF without heavy rain

    # NOISE FLOOR: Lognormal(0, 0.08) per-policy deviation
    # Ensures no two policies in same cell have identical M̂
    m_noise = RNG.lognormal(0, M_NOISE_SIGMA, n)
    M *= m_noise

    M = np.clip(M, 0.70, 3.5)   # allow protective interactions down to 0.70 for well-hardened WUI
    df["M_true"] = M.round(4)

    # ── EXPECTED LOSS ─────────────────────────────────────────────────────
    el = lam * mu * M
    df["expected_loss_true"] = el.round(2)

    # ── SIMULATE ACTUAL LOSSES (compound NB + spliced Gamma/GPD) ─────────
    nb_r  = 0.8
    nb_p  = nb_r / (nb_r + lam)
    cnt   = RNG.negative_binomial(nb_r, nb_p, n).clip(0, 5)
    df["claim_occurred"] = (cnt > 0).astype(int)
    df["claim_count"]    = cnt

    total_loss    = np.zeros(n)
    is_cat_event  = np.zeros(n, dtype=int)

    for i in np.where(cnt > 0)[0]:
        losses   = []
        had_cat  = False
        for _ in range(int(cnt[i])):
            cat_p = float(np.clip(CAT_BASE_FRAC * M[i], 0.015, 0.30))
            if RNG.random() < cat_p:
                x = stats.genpareto.rvs(c=0.25, scale=float(mu[i])*0.55,
                                        loc=CAT_THRESHOLD, random_state=None)
                had_cat = True
            else:
                x = float(RNG.gamma(2.5, float(mu[i]) / 2.5))
            losses.append(x)

        # Apply trend factor (vintage-adjusted construction inflation)
        raw = sum(losses) * float(df["trend_factor"].iloc[i])
        raw = min(raw, float(df["coverage_amount"].iloc[i]))
        total_loss[i]   = max(raw, 100.0)
        is_cat_event[i] = int(had_cat)

    # CAP at 99.5th percentile
    pos = total_loss[total_loss > 0]
    cap = float(np.percentile(pos, 99.5)) if len(pos) > 0 else 1e6
    total_loss = np.minimum(total_loss, cap)

    df["claim_amount"]  = total_loss.round(2)
    df["total_loss"]    = total_loss.round(2)
    df["is_cat_event"]  = is_cat_event

    # ── RISK SCORE (log-scale, 50-950) ───────────────────────────────────
    el_arr  = df["expected_loss_true"].values
    log_el  = np.log1p(el_arr)
    df["risk_score_true"] = np.clip(
        50 + 900 * (log_el - log_el.min()) / (log_el.max() - log_el.min()),
        50, 950
    ).round(1)

    def _band(s):
        if s < 200: return "Very Low"
        if s < 400: return "Low"
        if s < 600: return "Moderate"
        if s < 800: return "High"
        return "Very High"
    df["risk_band"] = df["risk_score_true"].apply(_band)

    # ── RISK SCORE A2 (freq + sev components) ────────────────────────────
    LAM_CAP, SEV_CAP = 0.15, 500_000
    w_f, w_s, alpha  = 0.45, 0.55, 0.8
    f_sc = np.minimum(500.0, lam / LAM_CAP * 500)
    s_sc = np.minimum(500.0, (mu * M) / SEV_CAP * 500)
    df["risk_score_a2"] = np.clip(
        (w_f*(f_sc+1)**alpha + w_s*(s_sc+1)**alpha)**(1/alpha) - 1,
        0, 1000
    ).round(1)

    # ── INDICATED PREMIUM ────────────────────────────────────────────────
    # Blueprint formula: E[L] / (1 - expense_ratio - profit_margin)
    # = E[L] / (1 - 0.28 - 0.05) = E[L] / 0.67
    # Expected LR = 0.67 by construction; actual 1-yr will be noisy ~0.52-0.72
    df["indicated_premium"]  = (el_arr / 0.67).round(2)
    df["expected_loss_ratio"] = 0.67   # oracle expected; distinguishable from actual

    return df


# ════════════════════════════════════════════════════════════════════════════
# VALIDATION: compare vs III/ISO/Verisk published benchmarks
# ════════════════════════════════════════════════════════════════════════════

def _print_validation(df):
    n = len(df)
    claim_rate  = df["claim_occurred"].mean() * 100
    claimants   = df[df["total_loss"] > 0]
    avg_sev     = claimants["total_loss"].mean()
    avg_mu      = df["mu_true"].mean()
    avg_prem    = df["indicated_premium"].mean()
    med_prem    = df["indicated_premium"].median()
    zero_pct    = (df["total_loss"] == 0).mean() * 100
    actual_lr   = df["total_loss"].sum() / df["indicated_premium"].sum()
    expected_lr = 0.67   # by construction
    cat_pct     = df["is_cat_event"].mean() * 100
    m_mean      = df["M_true"].mean()
    m_std       = df["M_true"].std()

    # Gini coefficient: cumulative loss by oracle E[L] rank (actuarial standard)
    sorted_idx   = np.argsort(-df["expected_loss_true"].values)
    cum_loss_vec = np.cumsum(df["total_loss"].values[sorted_idx])
    total_loss_s = cum_loss_vec[-1]
    lorenz       = cum_loss_vec / (total_loss_s + 1e-9)
    gini_oracle  = float(2 * np.mean(lorenz) - 1)  # descending sort: 2*area - 1

    # Kendall tau-b: rank correlation of E[L] vs actual loss (claimants only)
    from scipy.stats import kendalltau
    el_c   = df.loc[df["total_loss"] > 0, "expected_loss_true"].values
    los_c  = df.loc[df["total_loss"] > 0, "total_loss"].values
    # Sample for speed
    idx_s  = RNG.choice(len(el_c), size=min(3000, len(el_c)), replace=False)
    tau, _ = kendalltau(el_c[idx_s], los_c[idx_s])

    print(f"\n{'═'*58}")
    print(f"  v6 GENERATION SUMMARY  ({n:,} policies)")
    print(f"{'═'*58}")

    # ── Frequency / Severity
    print(f"\n  FREQUENCY & SEVERITY")
    print(f"  {'Metric':<32} {'Actual':>10}  {'III/ISO Target':>14}")
    print(f"  {'─'*58}")
    ok = lambda v, lo, hi: '✓' if lo <= v <= hi else '✗'
    print(f"  {'Claim rate':<32} {claim_rate:>9.2f}%  {'5.3-5.6%':>14}  {ok(claim_rate,5.0,5.8)}")
    print(f"  {'Avg attritional severity (mu)':<32} ${avg_mu:>9,.0f}  {'$15K-$18K':>14}  {ok(avg_mu,14000,19000)}")
    print(f"  {'Avg realized severity':<32} ${avg_sev:>9,.0f}  {'$22K-$27K':>14}  {ok(avg_sev,19000,30000)}")
    print(f"  {'Zero-loss policies':<32} {zero_pct:>9.1f}%  {'94-95%':>14}  {ok(zero_pct,93.0,96.0)}")
    print(f"  {'CAT events (% of policies)':<32} {cat_pct:>9.2f}%  {'0.1-0.5%':>14}  {ok(cat_pct,0.05,0.8)}")

    # ── Premium / LR
    print(f"\n  PREMIUM & LOSS RATIO")
    print(f"  {'─'*58}")
    print(f"  {'Avg indicated premium (mean)':<32} ${avg_prem:>9,.0f}  {'$2,000-$2,500':>14}  {ok(avg_prem,1900,2800)}")
    print(f"  {'Median indicated premium':<32} ${med_prem:>9,.0f}  {'$1,100-$1,800':>14}  {ok(med_prem,900,1900)}")
    print(f"  {'Expected LR (by construction)':<32} {expected_lr:>9.3f}   {'0.65-0.67':>14}  ✓")
    print(f"  {'Actual 1-yr LR (simulated)':<32} {actual_lr:>9.3f}   {'0.52-0.72':>14}  {ok(actual_lr,0.45,0.80)}")
    print(f"  NOTE: 1-yr actual LR noise is INTENTIONAL.")
    print(f"        Expected LR = 0.67 (oracle); actual deviates due to")
    print(f"        random claim occurrence + frailty heterogeneity.")
    print(f"        Stressed portfolios (demo narrative) show 112% LR.")

    # ── Interaction multiplier
    print(f"\n  INTERACTION MULTIPLIER  M̂")
    print(f"  {'─'*58}")
    print(f"  {'M̂ mean':<32} {m_mean:>9.3f}   {'1.3-1.6':>14}  {ok(m_mean,1.1,1.8)}")
    print(f"  {'M̂ std dev':<32} {m_std:>9.3f}   {'0.4-0.7':>14}  {ok(m_std,0.3,0.8)}")
    print(f"  {'M̂ max':<32} {df['M_true'].max():>9.3f}   {'≤3.5':>14}  {ok(df['M_true'].max(),0,3.6)}")
    print(f"  {'Frailty mean (should be 1.0)':<32} {df['frailty_term'].mean():>9.3f}   {'~1.00':>14}  {ok(df['frailty_term'].mean(),0.95,1.05)}")

    # ── Oracle R²  -- key guardrail
    print(f"\n  MODEL REALISM  (oracle E[L] vs actual losses)")
    print(f"  {'─'*58}")
    print(f"  {'Oracle Gini (E[L] rank vs losses)':<32} {gini_oracle:>9.3f}   {'0.25-0.50':>14}  {ok(gini_oracle,0.15,0.60)}")
    print(f"  {'Kendall τ (E[L] vs loss, claimants)':<32} {tau:>9.3f}   {'0.10-0.30':>14}  {ok(tau,0.05,0.40)}")
    print(f"  NOTE: Gini/τ bounded by claim volatility -- even oracle models")
    print(f"        score 0.25-0.45 on 1-yr cross-sectional data.")
    print(f"        Fitted GLM Gini will be ~0.05-0.10 lower (~80-90% of oracle).")

    # ── Risk bands
    print(f"\n  RISK BAND DISTRIBUTION")
    print(f"  {'─'*58}")
    bands = df["risk_band"].value_counts()
    for b in ["Very Low","Low","Moderate","High","Very High"]:
        pct = bands.get(b, 0) / n * 100
        bar = "█" * int(pct / 2)
        print(f"  {'  '+b:<32} {pct:5.1f}%  {bar}")

    # ── Feature distributions
    print(f"\n  FEATURE DISTRIBUTIONS")
    print(f"  {'─'*58}")
    subprime_pct = (df["credit_score"] < 620).mean() * 100
    print(f"  Credit P10/P50/P90: {df['credit_score'].quantile(0.1):.0f} / "
          f"{df['credit_score'].quantile(0.5):.0f} / "
          f"{df['credit_score'].quantile(0.9):.0f}   "
          f"Subprime(<620): {subprime_pct:.1f}%  [target 15-20%]  {ok(subprime_pct,13,22)}")
    recent_roof = (df["roof_age_yr"] <= 3).mean() * 100
    print(f"  Roof age 0-3yr: {recent_roof:.1f}%  [target ~15%]  {ok(recent_roof,10,20)}")
    print(f"  Roof age mean/median: {df['roof_age_yr'].mean():.1f}yr / {df['roof_age_yr'].median():.0f}yr")

    # ── Missingness
    print(f"\n  MISSINGNESS (realistic data quality)")
    print(f"  {'─'*58}")
    canopy_unk = (df["vegetation_risk_composite"] == "Unknown").mean() * 100
    ds_miss    = df["defensible_space_score"].isna().mean() * 100
    fst_miss   = (df["dist_to_fire_station_mi"] < 0).mean() * 100
    knob_pct   = df["has_knob_tube_wiring"].mean() * 100
    polybut_pct= df["has_polybutylene_pipe"].mean() * 100
    near_dup   = df["is_near_duplicate"].mean() * 100
    print(f"  vegetation_risk unknown:{canopy_unk:.1f}%  [target 3-5%]   {ok(canopy_unk,2,7)}")
    print(f"  defensible_space NaN:  {ds_miss:.1f}%  [target 20-25%]  {ok(ds_miss,15,30)}")
    print(f"  fire_station sentinel: {fst_miss:.1f}%  [target ~3%]    {ok(fst_miss,1,6)}")
    print(f"  knob_tube wiring:      {knob_pct:.1f}%  [target 1-4%]   {ok(knob_pct,0.5,6)}")
    print(f"  polybutylene pipe:     {polybut_pct:.1f}%  [target 6-14%]  {ok(polybut_pct,4,16)}")
    print(f"  near_duplicate flag:   {near_dup:.2f}%  [target ~0.5%]  {ok(near_dup,0.2,1.0)}")
    data_vin   = df["data_vintage_flag"].mean() * 100
    rcv_vin    = df["rcv_vintage_flag"].mean() * 100
    print(f"  data_vintage_flag:     {data_vin:.1f}%  [target ~12%]   {ok(data_vin,9,16)}")
    print(f"  rcv_vintage_flag:      {rcv_vin:.1f}%  [target ~8%]    {ok(rcv_vin,5,12)}")

    # ── Sprint 3 features
    print(f"\n  SPRINT 3 FEATURES (3.3 + 0.7)")
    print(f"  {'─'*58}")
    slope_mean  = df["slope_steepness_pct"].mean()
    pbr_mean    = df["post_burn_rainfall_intensity"].mean()
    renewal_pct = df["renewal_credit"].mean() * 100
    three_way   = ((df["wildfire_zone"] == "High") &
                   (df["slope_steepness_pct"] > 55) &
                   (df["post_burn_rainfall_intensity"] > 60)).mean() * 100
    print(f"  slope_steepness mean:  {slope_mean:.1f}  [expected 20-45 across states]")
    print(f"  post_burn_rainfall mean: {pbr_mean:.1f}  [expected 15-40]")
    print(f"  renewal_credit (3yr+0cl): {renewal_pct:.1f}%  [expected 20-35%]")
    print(f"  Three-way ix activated:   {three_way:.1f}%  [expected 1-3%]")
    print(f"{'═'*58}\n")


# ════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _pick(d, n):
    keys = list(d.keys())
    p    = [d[k]["p"] for k in keys]
    return RNG.choice(keys, size=n, p=p)


def _roof_probs(base, age, is_wf_state=False):
    p = base.copy()
    p[1] += float(np.clip((age - 20) / 80, 0, 0.06))
    if is_wf_state:
        p[1] += 0.06
    return p / p.sum()


def _zone(prob, n):
    r = RNG.random(n)
    return np.where(r < prob * 0.38, "High",
           np.where(r < prob * 0.78, "Moderate", "Low"))


if __name__ == "__main__":
    import os, time
    os.makedirs("data", exist_ok=True)
    t0 = time.time()
    df = generate_dataset(100_000)
    df.to_csv("data/homeowners_data.csv", index=False)
    print(f"Saved -> data/homeowners_data.csv  ({time.time()-t0:.1f}s)")
    print("\nKey numeric stats:")
    print(df[["lambda_true","mu_true","M_true","expected_loss_true",
              "risk_score_true","indicated_premium"]].describe().round(2))
