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

    lower_median = _find_median(ci[lower_col])   # lower survival CI → earlier crossing → lower median
    upper_median = _find_median(ci[upper_col])   # upper survival CI → later crossing → upper median
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


def build_km_supporting_table(
    cohort_df: pd.DataFrame,
    km_output: dict,
    time_points: list = None,
    time_col: str = "os_time_months",
    event_col: str = "os_event",
    group_col: str = "cohort",
) -> dict:
    """
    Build supporting tables for the KM survival curve:
      1. Summary table: N, events, median OS (95% CI), OS rates at fixed time points,
         median follow-up (reverse KM)
      2. Number-at-risk table at fixed time points

    Returns dict with keys: 'summary_table', 'number_at_risk'
    """
    if time_points is None:
        time_points = [0, 6, 12, 18, 24, 30, 36]

    km_results = km_output["km_results"]
    lr = km_output["logrank"]
    groups = sorted(km_results.keys())

    # ── Summary table ────────────────────────────────────────────────────
    summary_rows = []
    for grp in groups:
        kmf = km_results[grp]
        mask = cohort_df[group_col] == grp
        grp_df = cohort_df[mask]
        n = len(grp_df)
        n_events = int(grp_df[event_col].sum())

        # Median OS and 95% CI
        median_os = kmf.median_survival_time_
        try:
            from lifelines.utils import median_survival_times
            ci_df = median_survival_times(kmf.confidence_interval_survival_function_)
            ci_lower = float(ci_df.iloc[0, 0])
            ci_upper = float(ci_df.iloc[0, 1])
        except Exception:
            ci_lower, ci_upper = np.nan, np.nan

        # OS rates at fixed time points
        sf = kmf.survival_function_
        ci_sf = kmf.confidence_interval_survival_function_
        lower_col = ci_sf.columns[0]
        upper_col = ci_sf.columns[1]

        os_rates = {}
        for t in time_points:
            if t == 0:
                continue
            try:
                idx = sf.index[sf.index <= t]
                if len(idx) == 0:
                    rate = 1.0
                    rate_lo, rate_hi = 1.0, 1.0
                else:
                    rate = float(sf.loc[idx[-1]].iloc[0])
                    rate_lo = float(ci_sf.loc[idx[-1], lower_col])
                    rate_hi = float(ci_sf.loc[idx[-1], upper_col])
                os_rates[t] = f"{rate*100:.1f}% ({rate_lo*100:.1f}–{rate_hi*100:.1f})"
            except Exception:
                os_rates[t] = "—"

        # Median follow-up (reverse KM: flip event indicator)
        kmf_rev = KaplanMeierFitter()
        kmf_rev.fit(
            durations=grp_df[time_col],
            event_observed=1 - grp_df[event_col],  # reversed
        )
        median_fu = kmf_rev.median_survival_time_

        n_censored = n - n_events
        row = {
            "Cohort": grp,
            "N": n,
            "Events, n (%)": f"{n_events} ({n_events/n*100:.1f}%)",
            "Censored, n (%)": f"{n_censored} ({n_censored/n*100:.1f}%)",
            "Median follow-up, months": round(median_fu, 1),
            "Median OS, months (95% CI)": f"{median_os:.1f} ({ci_lower:.1f}-{ci_upper:.1f})" if not np.isnan(ci_lower) else f"{median_os:.1f} (NR-NR)",
        }
        for t in time_points:
            if t == 0:
                continue
            row[f"OS at {t}m, % (95% CI)"] = os_rates[t]
        summary_rows.append(row)

    # Add log-rank p-value as a footer row
    if lr is not None:
        footer = {"Cohort": f"Log-rank p-value: {lr.p_value:.4f}"}
        summary_rows.append(footer)

    summary_table = pd.DataFrame(summary_rows)

    # ── Number-at-risk table ─────────────────────────────────────────────
    nar_rows = []
    for grp in groups:
        kmf = km_results[grp]
        row = {"Cohort": grp}
        for t in time_points:
            try:
                et = kmf.event_table
                at_risk = et.loc[et.index <= t, "at_risk"]
                row[f"t={t}m"] = int(at_risk.iloc[-1]) if len(at_risk) > 0 else 0
            except Exception:
                row[f"t={t}m"] = "—"
        nar_rows.append(row)
    number_at_risk = pd.DataFrame(nar_rows)

    return {
        "summary_table": summary_table,
        "number_at_risk": number_at_risk,
    }


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
