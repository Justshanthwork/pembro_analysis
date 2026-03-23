"""
analysis.py — Landmark Survival Analysis (KM, Log-Rank, Cox PH)
================================================================
Implements the statistical methods from the SAP:
  - Kaplan-Meier survival estimation per cohort
  - Log-rank test comparing survival distributions
  - Multivariate Cox proportional hazards model
"""

import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test


def run_kaplan_meier(
    cohort_df: pd.DataFrame,
    time_col: str = "os_time_months",
    event_col: str = "os_event",
    group_col: str = "cohort",
) -> dict:
    """
    Fit KM curves for each cohort and run log-rank test.

    Returns
    -------
    dict with keys:
      - km_results: {group_name: KaplanMeierFitter}
      - logrank: logrank_test result object
      - summary: DataFrame with median OS and 95% CI per group
    """
    groups = cohort_df[group_col].unique()
    km_results = {}

    for grp in sorted(groups):
        mask = cohort_df[group_col] == grp
        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=cohort_df.loc[mask, time_col],
            event_observed=cohort_df.loc[mask, event_col],
            label=grp,
        )
        km_results[grp] = kmf

    # Log-rank test
    grp_names = sorted(groups)
    if len(grp_names) == 2:
        mask1 = cohort_df[group_col] == grp_names[0]
        mask2 = cohort_df[group_col] == grp_names[1]
        lr = logrank_test(
            durations_A=cohort_df.loc[mask1, time_col],
            durations_B=cohort_df.loc[mask2, time_col],
            event_observed_A=cohort_df.loc[mask1, event_col],
            event_observed_B=cohort_df.loc[mask2, event_col],
        )
    else:
        lr = None

    # Median survival summary
    summary_rows = []
    for grp, kmf in km_results.items():
        median_ci = kmf.confidence_interval_cumulative_density_
        median_surv = kmf.median_survival_time_
        # Get CI for median
        try:
            ci = median_confidence_interval(kmf)
        except Exception:
            ci = (np.nan, np.nan)
        n = (cohort_df[group_col] == grp).sum()
        n_events = cohort_df.loc[cohort_df[group_col] == grp, event_col].sum()
        summary_rows.append({
            "Cohort": grp,
            "N": n,
            "Events": int(n_events),
            "Median OS (months)": round(median_surv, 1),
            "95% CI Lower": round(ci[0], 1),
            "95% CI Upper": round(ci[1], 1),
        })

    summary = pd.DataFrame(summary_rows)

    return {
        "km_results": km_results,
        "logrank": lr,
        "summary": summary,
    }


def median_confidence_interval(kmf: KaplanMeierFitter) -> tuple[float, float]:
    """Extract 95% CI for median survival time from a KM fitter."""
    # Use lifelines built-in
    from lifelines.utils import median_survival_times
    ci = kmf.confidence_interval_survival_function_
    lower_col = ci.columns[0]
    upper_col = ci.columns[1]

    def _find_median(series):
        below = series[series <= 0.5]
        if below.empty:
            return np.nan
        return below.index[0]

    lower_median = _find_median(ci[upper_col])   # upper CI → lower median
    upper_median = _find_median(ci[lower_col])   # lower CI → upper median
    return (lower_median, upper_median)


def run_cox_model(
    cohort_df: pd.DataFrame,
    time_col: str = "os_time_months",
    event_col: str = "os_event",
    treatment_col: str = "cohort",
    covariates: list[str] | None = None,
) -> dict:
    """
    Fit multivariate Cox proportional hazards model.

    Parameters
    ----------
    cohort_df : analysis DataFrame
    covariates : list of covariate column names (if None, uses treatment only)

    Returns
    -------
    dict with keys:
      - cph: CoxPHFitter object
      - summary: DataFrame of HR, CI, p-values
      - concordance: C-index
    """
    # Prepare data
    df = cohort_df.copy()

    # Encode treatment as binary: Continuation=1, Fixed-Duration=0
    df["treatment_continuation"] = (df[treatment_col] == "Continuation").astype(int)

    # Determine which covariates to include
    model_cols = ["treatment_continuation"]

    if covariates:
        for cov in covariates:
            if cov not in df.columns:
                continue
            if df[cov].dtype == "object" or str(df[cov].dtype) == "category":
                # One-hot encode categorical variables
                dummies = pd.get_dummies(df[cov], prefix=cov, drop_first=True, dtype=int)
                df = pd.concat([df, dummies], axis=1)
                model_cols.extend(dummies.columns.tolist())
            else:
                # Numeric: fill NaN with median
                df[cov] = df[cov].fillna(df[cov].median())
                model_cols.append(cov)

    # Build model DataFrame
    model_df = df[model_cols + [time_col, event_col]].copy()
    model_df = model_df.dropna()

    # Ensure no zero or negative times
    model_df = model_df[model_df[time_col] > 0]

    # Fit Cox model
    cph = CoxPHFitter()
    cph.fit(
        model_df,
        duration_col=time_col,
        event_col=event_col,
        show_progress=False,
    )

    return {
        "cph": cph,
        "summary": cph.summary,
        "concordance": cph.concordance_index_,
    }


def print_km_summary(km_output: dict) -> None:
    """Print KM results to console."""
    print("\n" + "=" * 70)
    print("KAPLAN-MEIER SURVIVAL ANALYSIS — OS from 29-Month Landmark")
    print("=" * 70)
    print(km_output["summary"].to_string(index=False))

    if km_output["logrank"] is not None:
        lr = km_output["logrank"]
        print(f"\nLog-rank test:  χ² = {lr.test_statistic:.3f},  p = {lr.p_value:.4f}")
    print("=" * 70)


def print_cox_summary(cox_output: dict) -> None:
    """Print Cox model results to console."""
    print("\n" + "=" * 70)
    print("COX PROPORTIONAL HAZARDS MODEL")
    print("=" * 70)
    summary = cox_output["summary"][["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].copy()
    summary.columns = ["HR", "HR Lower 95%", "HR Upper 95%", "p-value"]
    print(summary.to_string())
    print(f"\nConcordance Index: {cox_output['concordance']:.3f}")
    print("=" * 70)
