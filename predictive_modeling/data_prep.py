"""
data_prep.py — Feature Engineering, Missingness Handling, and Imputation
=========================================================================
Transforms the analysis cohort (one row per patient from cohort_selection)
into a model-ready feature matrix with:

  1. Encoding of categorical variables (one-hot or ordinal as appropriate)
  2. Missingness indicators for high-missingness columns (PD-L1, histology)
  3. Multiple imputation via IterativeImputer for sensitivity analysis
  4. Binary treatment indicator and survival outcome extraction

Designed to be called from main.py or imported standalone.
"""

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder

from config import (
    ALL_FEATURES,
    TREATMENT_COL, TREATMENT_POSITIVE, TREATMENT_NEGATIVE,
    OUTCOME_EVENT_COL, OUTCOME_TIME_COL,
    HIGH_MISSINGNESS_COLS,
    N_IMPUTATIONS, IMPUTATION_MAX_ITER, IMPUTATION_SEED,
)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def prepare_features(
    cohort_df: pd.DataFrame,
    features: list[str] | None = None,
    add_missingness_indicators: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Convert the cohort DataFrame into a numeric feature matrix.

    Parameters
    ----------
    cohort_df : DataFrame from cohort_selection.select_cohort()
    features : list of column names to include (default: ALL_FEATURES)
    add_missingness_indicators : if True, add binary columns for high-miss features

    Returns
    -------
    X : DataFrame of numeric features (n_patients x n_features)
    T : Series of binary treatment (1 = Continuation, 0 = Fixed-Duration)
    Y_event : Series of event indicator (1 = death)
    Y_time : Series of survival time in months from landmark
    """
    if features is None:
        features = ALL_FEATURES

    # Only keep features that exist in the cohort
    available = [f for f in features if f in cohort_df.columns]
    missing_cols = set(features) - set(available)
    if missing_cols:
        print(f"  [data_prep] Warning: features not in cohort: {missing_cols}")

    df = cohort_df[available].copy()

    # ── 1. Treatment indicator ───────────────────────────────────────────
    T = (cohort_df[TREATMENT_COL] == TREATMENT_POSITIVE).astype(int)

    # ── 2. Outcomes ──────────────────────────────────────────────────────
    Y_event = cohort_df[OUTCOME_EVENT_COL].astype(int)
    Y_time = cohort_df[OUTCOME_TIME_COL].astype(float)

    # ── 3. Missingness indicators ────────────────────────────────────────
    if add_missingness_indicators:
        for col in HIGH_MISSINGNESS_COLS:
            if col in df.columns:
                indicator_name = f"{col}_missing"
                df[indicator_name] = _is_missing_or_unknown(df[col]).astype(int)

    # ── 4. Encode features ───────────────────────────────────────────────
    X = _encode_all_features(df)

    return X, T, Y_event, Y_time


def create_imputed_datasets(
    X: pd.DataFrame,
    n_imputations: int = N_IMPUTATIONS,
    max_iter: int = IMPUTATION_MAX_ITER,
    seed: int = IMPUTATION_SEED,
) -> list[pd.DataFrame]:
    """
    Generate multiple imputed versions of the feature matrix.

    Uses sklearn's IterativeImputer (MICE-style). Each imputed dataset
    can be used independently for model fitting, with results pooled
    afterwards (Rubin's rules).

    Parameters
    ----------
    X : Feature matrix with NaN for missing values
    n_imputations : number of imputed datasets to create
    max_iter : max iterations per imputation
    seed : base random seed (incremented per imputation)

    Returns
    -------
    List of DataFrames, each a complete (no-NaN) version of X
    """
    imputed_datasets = []

    for i in range(n_imputations):
        imputer = IterativeImputer(
            max_iter=max_iter,
            random_state=seed + i,
            sample_posterior=True,  # adds stochastic variation across imputations
        )
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index,
        )
        imputed_datasets.append(X_imputed)
        print(f"  [imputation] Dataset {i + 1}/{n_imputations} complete")

    return imputed_datasets


def summarize_missingness(X: pd.DataFrame) -> pd.DataFrame:
    """Return a summary of missing values per feature."""
    n = len(X)
    miss_count = X.isna().sum()
    miss_pct = (miss_count / n * 100).round(1)
    summary = pd.DataFrame({
        "feature": X.columns,
        "n_missing": miss_count.values,
        "pct_missing": miss_pct.values,
    }).sort_values("pct_missing", ascending=False).reset_index(drop=True)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_missing_or_unknown(series: pd.Series) -> pd.Series:
    """Detect missing values including string 'Unknown', 'unknown', NaN."""
    is_na = series.isna()
    is_unknown = series.astype(str).str.strip().str.lower().isin(
        ["unknown", "nan", "none", "", "not tested", "test failed",
         "tested - result unknown", "not tested",
         "test not performed due to insufficient sample",
         "test failed due to insufficient sample"]
    )
    return is_na | is_unknown


def _encode_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode all features to numeric representation.

    Strategy:
      - Continuous (age_at_index): keep as-is
      - Ordinal (ecog_ps): map to integer scale
      - Binary yes/no: map to 0/1
      - Nominal categorical: one-hot encode

    Unknown / missing values are set to NaN so the imputer can handle them.
    """
    result = pd.DataFrame(index=df.index)

    for col in df.columns:
        if col.endswith("_missing"):
            # Already binary indicator, pass through
            result[col] = df[col]
            continue

        series = df[col].copy()

        # ── Continuous features ──────────────────────────────────────
        if col == "age_at_index":
            result[col] = pd.to_numeric(series, errors="coerce")
            continue

        # ── Ordinal: ECOG performance status ─────────────────────────
        if col == "ecog_ps":
            ecog_map = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4}
            result[col] = series.astype(str).str.strip().map(ecog_map)
            # Unknown -> NaN (handled by imputer)
            continue

        # ── Binary yes/no features ───────────────────────────────────
        if col in ("brain_mets", "diabetes", "chronic_respiratory_disease",
                    "severe_cardiac", "chronic_kidney_disease",
                    "beta_blocker", "diuretic", "painkiller", "steroid",
                    "rasi", "anticoagulant"):
            result[col] = series.map({"Yes": 1, "No": 0, "yes": 1, "no": 0})
            # Unknown -> NaN
            continue

        # ── Binary two-level features ────────────────────────────────
        if col == "gender":
            result[col] = series.map({"M": 0, "F": 1, "MALE": 0, "FEMALE": 1})
            continue

        if col == "de_novo_vs_recurrent":
            result[col] = series.map({"De novo": 0, "Recurrent": 1})
            continue

        if col == "pembro_with_chemo":
            result[col] = series.map({"Monotherapy": 0, "With Chemo": 1})
            continue

        # ── Nominal categoricals: one-hot encode ─────────────────────
        if col in ("race", "payer", "smoking_history", "pdl1_status", "histology"):
            # Replace unknowns with NaN before encoding
            cleaned = series.copy()
            cleaned[_is_missing_or_unknown(cleaned)] = np.nan

            # One-hot encode known categories
            dummies = pd.get_dummies(cleaned, prefix=col, dummy_na=False)
            # Drop most frequent category to avoid multicollinearity
            if len(dummies.columns) > 1:
                # Find the category with most observations and drop it
                drop_col = dummies.sum().idxmax()
                dummies = dummies.drop(columns=[drop_col])

            # Where original was NaN, set all dummies to NaN (for imputation)
            nan_mask = cleaned.isna()
            dummies[nan_mask] = np.nan

            result = pd.concat([result, dummies], axis=1)
            continue

        # ── Fallback: label-encode ───────────────────────────────────
        le = LabelEncoder()
        valid_mask = series.notna() & ~_is_missing_or_unknown(series)
        encoded = pd.Series(np.nan, index=series.index, dtype=float)
        if valid_mask.any():
            encoded[valid_mask] = le.fit_transform(series[valid_mask].astype(str))
        result[col] = encoded

    return result
