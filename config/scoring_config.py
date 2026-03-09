"""
config/scoring_config.py
========================
Central configuration for all scoring constants, tier thresholds, and
interaction multipliers. Edit here to propagate changes across the app
and predictor without touching model or app code directly.
"""

# ── Pricing / loss ratio targets ───────────────────────────────────────────────
TARGET_LOSS_RATIO   = 0.65   # Well-performing HO book baseline
EXPENSE_LOAD_FACTOR = 1.18   # Expense loading multiplier (18% of PP)
EXPENSE_RATIO       = 0.28   # Component breakdown: 28% expense
PROFIT_MARGIN       = 0.05   # Component breakdown: 5% target profit
COMBINED_DIVISOR    = 0.67   # 1 - 0.28 - 0.05

AVG_PREMIUM_NATIONAL = 2_200  # $2,200 national average (NAIC 2023)

# ── Risk score tiers (0–1000 scale) ──────────────────────────────────────────
TIERS = {
    "Very Low":  (0,   200),
    "Low":       (200, 400),
    "Moderate":  (400, 600),
    "High":      (600, 800),
    "Very High": (800, 1001),
}

TIER_COLORS = {
    "Very Low":  "#22c55e",   # green
    "Low":       "#86efac",   # light green
    "Moderate":  "#facc15",   # amber
    "High":      "#f97316",   # orange
    "Very High": "#ef4444",   # red
}

TIER_ACTIONS = {
    "Very Low":  "Auto-accept — best terms",
    "Low":       "Auto-accept — standard terms",
    "Moderate":  "Accept with conditions",
    "High":      "Refer to senior underwriter",
    "Very High": "Decline / surplus lines only",
}

# ── M̂ Interaction multipliers ─────────────────────────────────────────────────
# Changing these propagates to predictor.py DEFAULT_PRICING_CFG automatically
# if predictor imports from here. Values are kept in sync manually for now.
M_OVERRIDES = {
    # Hazard amplifiers (upward)
    "wood_wf_high":        3.50,   # Wood Shake × High WF     — ember ignition
    "wood_wf_mod":         2.10,   # Wood Shake × Moderate WF
    "flood_coast":         2.20,   # High Flood × <5mi coast  — surge compound
    "nonwood_wf":          1.80,   # Non-wood × High WF
    "eq_high":             1.50,   # High earthquake zone
    "old_frame":           1.35,   # Old roof (>20yr) × Frame
    "frame_pc_high":       1.18,   # Frame × PC≥8
    "knob_tube_fire":      1.20,   # Knob-and-tube wiring
    "polybutylene_water":  1.22,   # Polybutylene pipe
    "slope_burn_rain":     1.45,   # Slope>55% × High WF × Rain>60 (Montecito)
    "slope_burn_only":     1.18,   # Slope>55% × High WF (no rain)
    # Protective mitigants (downward)
    "metal_wf_sprinkler":  0.82,   # Metal roof + sprinkler in WF zone
    "defensible_high":     0.88,   # High defensible space (≥80) in WUI
    "new_superior":        0.85,   # New (≤5yr) Superior construction
}

# ── Benchmark targets (III / ISO / Verisk 2022-2024) ─────────────────────────
BENCHMARKS = {
    "annual_claim_rate":         0.053,    # 5.3% per year (ISO 2023)
    "avg_severity_all":         18_000,    # $18K blended
    "avg_severity_fire":        86_000,    # $84K–$88K fire/lightning
    "avg_severity_wind":        14_700,    # Wind/hail
    "avg_severity_water":       15_400,    # Water damage
    "zero_claim_pct":            0.945,    # 94.5% policies zero in any year
    "target_loss_ratio":          0.63,    # 60–65% well-performing book
    "tweedie_power_p":             1.65,   # Compound Poisson-Gamma
}

# ── Demo property expected scores (for presenter reference) ──────────────────
DEMO_EXPECTED_SCORES = {
    "Austin TX":      614,   # Non-Standard — aging roof, water claims
    "Paradise CA":    820,   # Very High — Wood Shake × High WF (M̂=3.5)
    "Boulder CO":     380,   # Low — Metal roof neutralises WF
    "Houston TX":     590,   # Moderate — Flood × old frame
    "Montecito CA":   710,   # High — Slope × Burn × Rain
    "Naperville IL":  520,   # Moderate — hail corridor + aging roof
    "Miami FL":       680,   # High — wind + coastal surge + RCV gap
    "Portland OR":    550,   # Moderate — canopy + water history
    "Minneapolis MN": 180,   # Very Low — preferred baseline
    "Phoenix AZ":     490,   # Moderate — urban heat + WF fringe
}

# ── Financial impact defaults (ROI calculator) ────────────────────────────────
ROI_DEFAULTS = {
    "book_size_m":        250,    # $250M written premium
    "current_loss_ratio": 112,    # Legacy baseline (%)
    "lr_improvement_pts":  14,    # 14-pt improvement
    "expected_profit_m":   35,    # $35M annual swing
}
