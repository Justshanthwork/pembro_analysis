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
# 1. DATA PATHS  (change DATA_DIR on your work laptop)
# ─────────────────────────────────────────────────────────────────────────────
# Set via environment variable or fall back to a default
DATA_DIR = Path(os.environ.get(
    "PEMBRO_DATA_DIR",
    Path(__file__).resolve().parent / "data"   # default: ./data subfolder
))

OUTPUT_DIR = Path(os.environ.get(
    "PEMBRO_OUTPUT_DIR",
    Path(__file__).resolve().parent / "output"
))

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
    "labs":             f"{FILE_PREFIX}LABS{FILE_SUFFIX}",
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
COVARIATES = [
    "age_at_index",
    "gender",
    "race",
    "payer",
    "smoking_history",
    "ecog_ps",            # from riskscores table
    "pdl1_status",        # from biomarker table
    "histology",          # from disease table
    "de_novo_vs_recurrent",
    "brain_mets",         # from metastases table
    "pembro_with_chemo",  # from LOT regimen
]

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
USE_SYNTHETIC_IF_MISSING = True   # auto-generate test data if CSVs not found
SYNTHETIC_N_PATIENTS = 500
SYNTHETIC_SEED = 42
