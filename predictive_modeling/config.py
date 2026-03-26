"""
config.py — Configuration for Treatment Effect Heterogeneity Analysis
=====================================================================
Extends the parent pipeline's cohort to detect subgroups where
fixed-duration vs continuation pembrolizumab may differ in OS benefit.

All paths are relative so this module works from any working directory.
"""

import sys
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
# Parent pipeline lives one level up
PARENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT_DIR))

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature Sets ─────────────────────────────────────────────────────────────
# These must match column names produced by cohort_selection.select_cohort()

FEATURES_CORE = [
    "age_at_index",
    "gender",
    "race",
    "payer",
    "smoking_history",
    "ecog_ps",
    "pdl1_status",
    "histology",
    "de_novo_vs_recurrent",
    "brain_mets",
    "pembro_with_chemo",
]

FEATURES_COMORBIDITIES = [
    "diabetes",
    "chronic_respiratory_disease",
    "severe_cardiac",
    "chronic_kidney_disease",
]

FEATURES_MEDICATIONS = [
    "beta_blocker",
    "diuretic",
    "painkiller",
    "steroid",
    "rasi",
    "anticoagulant",
]

ALL_FEATURES = FEATURES_CORE + FEATURES_COMORBIDITIES + FEATURES_MEDICATIONS

# ── Treatment / Outcome Columns ─────────────────────────────────────────────
TREATMENT_COL = "cohort"               # "Fixed-Duration" vs "Continuation"
TREATMENT_POSITIVE = "Continuation"    # coded as T=1 in causal forest
TREATMENT_NEGATIVE = "Fixed-Duration"  # coded as T=0

OUTCOME_EVENT_COL = "os_event"         # 1 = death, 0 = censored
OUTCOME_TIME_COL = "os_time_months"    # months from landmark

# ── Causal Forest Settings ───────────────────────────────────────────────────
CAUSAL_FOREST_N_ESTIMATORS = 1000
CAUSAL_FOREST_MIN_SAMPLES_LEAF = 30     # clinical: need enough per leaf
CAUSAL_FOREST_MAX_DEPTH = None          # let the forest decide
CAUSAL_FOREST_CRITERION = "het"         # heterogeneity-based splitting
CAUSAL_FOREST_CV_FOLDS = 5
CAUSAL_FOREST_SEED = 42

# ── Risk Stratification Settings ────────────────────────────────────────────
RISK_N_GROUPS = 3                       # low / medium / high risk
RISK_MODEL_N_ESTIMATORS = 500
RISK_MODEL_SEED = 42

# ── Multiple Imputation Settings ────────────────────────────────────────────
N_IMPUTATIONS = 5                       # number of imputed datasets
IMPUTATION_MAX_ITER = 10
IMPUTATION_SEED = 42

# High-missingness columns that get missingness indicators
HIGH_MISSINGNESS_COLS = ["pdl1_status", "histology"]

# ── Visualization ────────────────────────────────────────────────────────────
FIGURE_DPI = 150
FIGURE_FORMAT = "png"
