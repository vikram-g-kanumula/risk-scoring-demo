# This directory holds trained model artifacts (created by setup.py).
# The file is excluded from git via .gitignore due to its size (~10MB).
#
# After running `python setup.py`, you will find:
#   artifacts.pkl   — Serialised GLM, ensemble models, encoders, and metrics
#
# artifacts.pkl contains:
#   glm          : sklearn TweedieRegressor (p=1.65)
#   glm_scaler   : StandardScaler for GLM features
#   glm_enc      : LabelEncoder for categorical features
#   freq_model   : HistGradientBoostingRegressor (lambda)
#   sev_model    : HistGradientBoostingRegressor (mu)
#   m_rf         : RandomForestRegressor (M-hat base learner)
#   m_hgb        : HistGradientBoostingRegressor (M-hat base learner)
#   m_et         : ExtraTreesRegressor (M-hat base learner)
#   m_meta_ridge : Ridge (M-hat level-1 meta-learner)
#   m_meta_iso   : IsotonicRegression (M-hat final calibration)
#   t3_features  : list[str]  — Tier 3 feature names
#   t12_features : list[str]  — Tier 1+2 feature names
#   metrics      : dict       — Training metrics summary
#   el_min/max   : float      — Portfolio E[L] bounds (for risk score normalisation)
