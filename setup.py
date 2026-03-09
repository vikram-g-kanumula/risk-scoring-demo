"""
setup.py  v3
============
One-command setup: generates synthetic data and trains all models.
Run ONCE before launching the Streamlit app — and again any time
model_trainer.py or data_generator.py changes.

  python setup.py

What this script does:
  1. Generates 100,000 synthetic homeowner policies (Gaussian copula,
     6 Tier-3 interactions, Sprint 3 variables)
  2. Splits 80/20 into train and test sets (persisted for reproducibility)
  3. Trains the full v3 model stack and writes models/artifacts.pkl

Estimated runtime:
  Data generation  ~60–90 seconds
  Model training   ~5–10 minutes  (Poisson + Gamma GLMs + M̂ ensemble)
  Total            ~6–11 minutes
"""

import os, sys, time

os.makedirs("data",   exist_ok=True)
os.makedirs("models", exist_ok=True)

print("=" * 70)
print("  US HOMEOWNERS RISK SCORING ENGINE  —  SETUP  v3")
print("  Architecture: Poisson GLM × Gamma GLM baseline | M̂ on GLM residuals")
print("=" * 70)

# ── Step 1: Synthetic data generation ─────────────────────────────────────────
t0 = time.time()
print("\n[1/3] Generating 100,000 synthetic policies …")
print("      (Gaussian copula, bimodal distributions, 6 Tier-3 interactions,")
print("       Sprint 3 variables: slope, post-burn rainfall, vintage flags)")

from data_generator import generate_dataset
df = generate_dataset(100_000)

csv_path   = "data/homeowners_data.csv"
train_path = "data/train_data.csv"
test_path  = "data/test_data.csv"

df.to_csv(csv_path, index=False)
print(f"      ✓ {len(df):,} policies saved → {csv_path}  ({time.time()-t0:.1f}s)")

# ── Step 2: Train/test split (persisted for reproducibility) ──────────────────
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path,   index=False)
print(f"      ✓ Train: {len(train_df):,} | Test: {len(test_df):,} rows split saved")

# ── Step 3: Model training ─────────────────────────────────────────────────────
t1 = time.time()
print("\n[2/3] Training v3 model stack …")
print("      GLM layer  : PoissonRegressor (λ) + GammaRegressor (μ) on T1+T2 features")
print("                   TweedieRegressor (p=1.65) — DOI rate-filing display artifact")
print("      M̂ layer    : GLM residuals → M_actual = claim / GLM_prediction")
print("                   RF + HistGBM + ExtraTrees → Ridge + Isotonic meta-learner")
print("                   Training target: M_actual (not oracle M_true)")

from model_trainer import train_all
arts = train_all(df)

# ── Step 4: Validation summary ─────────────────────────────────────────────────
t2 = time.time()
print("\n[3/3] Validation …")

print("\n" + "=" * 70)
print("  SETUP COMPLETE  ✓")
print("=" * 70)
print(f"\n  Data generation : {t1-t0:5.1f}s")
print(f"  Model training  : {t2-t1:5.1f}s")
print(f"  Total           : {t2-t0:5.1f}s")

print("\n  Model metrics:")
for k, v in arts["metrics"].items():
    print(f"    {k:<32s}: {v}")

print("\n  Key artifacts in models/artifacts.pkl:")
print("    poisson_glm    — PoissonRegressor (λ, T1+T2, log-link)")
print("    gamma_glm      — GammaRegressor   (μ, T1+T2, log-link, claimants only)")
print("    glm_scaler     — StandardScaler shared by both GLMs")
print("    tweedie_glm    — TweedieRegressor (p=1.65, DOI rate-filing display)")
print("    rf_m / hist_m / et_m  — M̂ base learners (T3 features, trained on M_actual)")
print("    ridge_meta / iso_meta — M̂ stacked meta-learner")
print("    glm_relativities      — exp(β) table for rate-filing display")
print("    metrics               — R², MAE, reclassification stats")

print("\n  Files written:")
print("    models/artifacts.pkl")
print("    data/homeowners_data.csv")
print("    data/train_data.csv")
print("    data/test_data.csv")

print("\n  Next step — launch the demo:")
print("    streamlit run app.py")
print("=" * 70)