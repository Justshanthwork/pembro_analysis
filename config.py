"""
config.py — Configuration for Pembrolizumab Fixed-Duration vs Continuation OS Analysis
=======================================================================================
SAP: Overall Survival After Fixed Duration vs Continuation of Pembrolizumab
         of 2 Years in First-Line Metastatic NSCLC: A Landmark Analysis

Modify DATA_DIR to point to the folder containing IC PrecisionQ CSV exports.
All other parameters derive from the SAP.
"""

from pathlib import Path
import os

# ─────────────────────────────────────────────────────────────────────────────
# 0. SNOWFLAKE CONNECTION  (used when CSVs are not available locally)
# ─────────────────────────────────────────────────────────────────────────────
SNOWFLAKE_CONFIG = {
    "user":          "prashanth.jain@integraconnect.com",
    "account":       "integraconnect-heor",
    "warehouse":     "WH_M",
    "database":      "DEV_SS_OUTPUT",
    "schema":        "PUBLIC",
    "role":          "RBA_NP-DATAANALYST",
    "authenticator": "externalbrowser",
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA PATHS  (change DATA_DIR on your work laptop)
# ─────────────────────────────────────────────────────────────────────────────
# Set via environment variable or fall back to the data folder below
DATA_DIR = Path(os.environ.get(
    "PEMBRO_DATA_DIR",
    r"C:\Users\prashanth.jain\Desktop\Projects\PEMBRO NSCLC\data\March 2026 DRAFT"
))

OUTPUT_DIR = Path(os.environ.get(
    "PEMBRO_OUTPUT_DIR",
    Path(__file__).resolve().parent / "output"
))

# Local parquet cache — tables are saved here after Snowflake pull
# Delete individual .parquet files (or run with --refresh) to force re-pull
CACHE_DIR = Path(os.environ.get(
    "PEMBRO_CACHE_DIR",
    Path(__file__).resolve().parent / "cache"
))

# Auto-invalidate cache after this many days (0 = never auto-expire)
CACHE_MAX_AGE_DAYS = 7

# ─────────────────────────────────────────────────────────────────────────────
# 2. FILE REGISTRY  (IC PrecisionQ NSCLC file names)
# ─────────────────────────────────────────────────────────────────────────────
FILE_PREFIX = "IC_PRECISIONQ_STN_DEATH_NSCLC_"
FILE_SUFFIX = "_20260310_190335.csv"

FILES = {
    "demographics":     f"{FILE_PREFIX}DEMOGRAPHICS{FILE_SUFFIX}",
    "disease":          f"{FILE_PREFIX}DISEASE{FILE_SUFFIX}",
    "lot":              f"{FILE_PREFIX}LOT{FILE_SUFFIX}",
    "dose":             f"{FILE_PREFIX}DOSE{FILE_SUFFIX}",
    "labs":             f"{FILE_PREFIX}MEASUREMENT{FILE_SUFFIX}",  # table is MEASUREMENT, not LABS
    "measurement":      f"{FILE_PREFIX}MEASUREMENT{FILE_SUFFIX}",
    "vitals":           f"{FILE_PREFIX}VITALS{FILE_SUFFIX}",
    "biomarker":        f"{FILE_PREFIX}BIOMARKER{FILE_SUFFIX}",
    "staginghistory":   f"{FILE_PREFIX}STAGINGHISTORY{FILE_SUFFIX}",
    "lotresponse":      f"{FILE_PREFIX}LOTRESPONSE{FILE_SUFFIX}",
    "metastases":       f"{FILE_PREFIX}METASTASES{FILE_SUFFIX}",
    "comorbidities":    f"{FILE_PREFIX}COMORBIDITIES{FILE_SUFFIX}",
    "procedure":        f"{FILE_PREFIX}PROCEDURE{FILE_SUFFIX}",
    "biopsy":           f"{FILE_PREFIX}BIOPSY{FILE_SUFFIX}",
    "riskscores":       f"{FILE_PREFIX}RISKSCORES{FILE_SUFFIX}",
    "diseasehistory":   f"{FILE_PREFIX}DISEASEHISTORY{FILE_SUFFIX}",
    "medicalcondition": f"{FILE_PREFIX}MEDICALCONDITION{FILE_SUFFIX}",
    "drugmodifications": f"{FILE_PREFIX}DRUGMODIFICATIONS{FILE_SUFFIX}",
    "visithistory":     f"{FILE_PREFIX}VISITHISTORY{FILE_SUFFIX}",
    "futurevisits":     f"{FILE_PREFIX}FUTUREVISITS{FILE_SUFFIX}",
    "cohort":           f"{FILE_PREFIX}COHORT{FILE_SUFFIX}",
    "patientcuration":  f"{FILE_PREFIX}PATIENTCURATION{FILE_SUFFIX}",
    "regimen":          f"{FILE_PREFIX}REGIMEN{FILE_SUFFIX}",
    "rsi":              f"{FILE_PREFIX}RSI{FILE_SUFFIX}",
    "controltable":     f"{FILE_PREFIX}CONTROLTABLE{FILE_SUFFIX}",
    "division":         f"{FILE_PREFIX}DIVISION{FILE_SUFFIX}",
    "remappedmpiid":    f"{FILE_PREFIX}REMAPPEDMPIID{FILE_SUFFIX}",
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. SAP PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

# Diagnosis date window
DIAGNOSIS_START = "2016-01-01"
DIAGNOSIS_END   = "2025-08-31"

# Follow-up cutoff
FOLLOWUP_CUTOFF = "2026-02-28"

# Pembrolizumab identification
PEMBRO_GENERIC_NAMES = ["pembrolizumab"]
PEMBRO_BRAND_NAMES   = ["keytruda"]

# Fixed-duration window: last infusion between 22–26 months from LOT start
FIXED_DURATION_LOWER_MONTHS = 22
FIXED_DURATION_UPPER_MONTHS = 26

# Continuation cohort: treatment >26 months
CONTINUATION_LOWER_MONTHS = 26

# Gap rule: if two consecutive infusions are >6 months apart,
# treatment is considered ended at the earlier date
MAX_INFUSION_GAP_DAYS = 183  # ~6 months

# Landmark time: 29 months from pembrolizumab initiation
LANDMARK_MONTHS = 29

# Exclusion: chemo within 2 months of last pembro infusion → progression stop
CHEMO_PROXIMITY_EXCLUSION_DAYS = 60  # 2 months

# ─────────────────────────────────────────────────────────────────────────────
# 4. COVARIATES for Cox model (evaluated at index date)
# ─────────────────────────────────────────────────────────────────────────────

# ── Recommended adjusted model (per collaborator feedback) ────────────────────
# Covariates: age (continuous), race, ECOG (≤1 vs 2+),
#             PD-L1 (<1%/Neg, 1-49%, ≥50%, Unknown),
#             histology (Squamous, Non-Squamous, Unknown),
#             treatment type (with/without chemo)
COVARIATES_ADJUSTED = [
    "age_at_index",      # continuous
    "race",              # categorical
    "ecog_binary",       # ≤1 vs 2+
    "pdl1_cat",          # <1%/Negative | 1-49% | ≥50% | Unknown
    "histology_cat",     # Squamous | Non-Squamous | Unknown
    "pembro_with_chemo", # With Chemo | Monotherapy
]

# Core covariates (broader set for sensitivity / full model)
COVARIATES_CORE = [
    "age_at_index",
    "gender",
    "race",
    "payer",
    "smoking_history",
    "ecog_ps",            # from vitals/labs/riskscores table
    "pdl1_status",        # from biomarker table
    "histology",          # from disease table
    "de_novo_vs_recurrent",
    "brain_mets",         # from metastases table
    "pembro_with_chemo",  # from LOT regimen
]

# Extended covariates (from comorbidities + medicalcondition tables)
# Inspired by ATHENA paper (Rousseau et al. Lancet Reg Health Eur 2024)
COVARIATES_COMORBIDITIES = [
    "diabetes",                     # from comorbidities (ICD E11)
    "chronic_respiratory_disease",  # from comorbidities (ICD J44)
    "severe_cardiac",               # MI, heart failure, CVD combined
    "chronic_kidney_disease",       # from comorbidities (ICD N18)
]

COVARIATES_MEDICATIONS = [
    "beta_blocker",        # from medicalcondition
    "diuretic",            # from medicalcondition
    "painkiller",          # from medicalcondition
    "steroid",             # from medicalcondition
    "rasi",                # renin-angiotensin system inhibitor
    "anticoagulant",       # from medicalcondition
]

# Full covariate set = core + comorbidities + medications
COVARIATES = COVARIATES_CORE + COVARIATES_COMORBIDITIES + COVARIATES_MEDICATIONS

# Covariate sets for multiple Cox models
# "adjusted" = primary recommended model; others for sensitivity
COX_MODEL_SPECS = {
    "unadjusted":  [],
    "adjusted":    COVARIATES_ADJUSTED,   # PRIMARY — per collaborator recommendation
    "full":        COVARIATES,            # sensitivity — all available covariates
}

# ─────────────────────────────────────────────────────────────────────────────
# 4b. SUBGROUP ANALYSES
# ─────────────────────────────────────────────────────────────────────────────
SUBGROUP_VARIABLES = [
    ("gender",              ["M", "F"]),
    ("histology_cat",       ["Squamous", "Non-Squamous"]),          # recoded per recommendation
    ("pdl1_cat",            ["≥50%", "1-49%", "<1% / Negative"]),   # recoded per recommendation
    ("pembro_with_chemo",   ["With Chemo", "Monotherapy"]),
    ("ecog_binary",         ["≤1", "2+"]),                          # recoded per recommendation
    ("brain_mets",          ["Yes", "No"]),
    ("de_novo_vs_recurrent", ["De novo", "Recurrent"]),
]

# ─────────────────────────────────────────────────────────────────────────────
# 4c. LANDMARK SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
LANDMARK_SENSITIVITY_MONTHS = [27, 29, 32]

# ─────────────────────────────────────────────────────────────────────────────
# 5. CHEMOTHERAPY AGENTS (to identify chemo component in regimens)
# ─────────────────────────────────────────────────────────────────────────────
CHEMO_AGENTS = [
    "carboplatin", "cisplatin", "pemetrexed", "paclitaxel",
    "nab-paclitaxel", "gemcitabine", "docetaxel", "etoposide",
    "vinorelbine",
]

# ─────────────────────────────────────────────────────────────────────────────
# 6. SYNTHETIC DATA SETTINGS (for testing when real data not available)
# ─────────────────────────────────────────────────────────────────────────────
USE_SYNTHETIC_IF_MISSING = False  # set True to test with synthetic data; False for real CSVs only
USE_LASSO_SELECTION = True      # use LASSO to select covariates for one model
LASSO_ALPHA_RANGE = [0.01, 0.05, 0.1, 0.5, 1.0]

SYNTHETIC_N_PATIENTS = 500
SYNTHETIC_SEED = 42
