"""
causal_forest.py — Causal Forest for Treatment Effect Heterogeneity
====================================================================
Estimates individualized treatment effects (ITEs) for continuation vs
fixed-duration pembrolizumab using the econml CausalForestDML estimator.

Key outputs:
  - Individual treatment effect estimates (CATE) for each patient
  - Feature importance for heterogeneity drivers
  - Subgroup-level treatment effects with confidence intervals
  - Calibration test for treatment effect heterogeneity

Clinical context:
  The parent pipeline found no average OS difference (log-rank p=0.47).
  This module asks: are there patient subgroups where the treatment
  effect is meaningfully positive or negative, even though the average
  is null?
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold

from config import (
    CAUSAL_FOREST_N_ESTIMATORS,
    CAUSAL_FOREST_MIN_SAMPLES_LEAF,
    CAUSAL_FOREST_MAX_DEPTH,
    CAUSAL_FOREST_CRITERION,
    CAUSAL_FOREST_CV_FOLDS,
    CAUSAL_FOREST_SEED,
)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def fit_causal_forest(
    X: pd.DataFrame,
    T: pd.Series,
    Y: pd.Series,
    Y_time: pd.Series | None = None,
    n_estimators: int = CAUSAL_FOREST_N_ESTIMATORS,
    min_samples_leaf: int = CAUSAL_FOREST_MIN_SAMPLES_LEAF,
    max_depth: int | None = CAUSAL_FOREST_MAX_DEPTH,
    criterion: str = CAUSAL_FOREST_CRITERION,
    seed: int = CAUSAL_FOREST_SEED,
) -> dict:
    """
    Fit a CausalForestDML to estimate conditional average treatment effects.

    We use Y = os_event (binary: death indicator) as the outcome.
    A negative CATE means continuation REDUCES mortality (beneficial).
    A positive CATE means continuation INCREASES mortality (harmful).

    The DML (double machine learning) framework uses nuisance models to
    partial out the effects of confounders on both treatment assignment
    and outcome, then estimates the residualized treatment effect.

    Parameters
    ----------
    X : Feature matrix (n_patients x n_features), numeric, may contain NaN
    T : Treatment indicator (1 = Continuation, 0 = Fixed-Duration)
    Y : Outcome (1 = death event, 0 = censored)
    Y_time : Survival time (not used directly in CATE, but stored for
             downstream weighted analyses)
    n_estimators : number of trees in the causal forest
    min_samples_leaf : minimum patients per leaf (set high for stability)
    max_depth : max tree depth (None = unlimited)
    criterion : splitting criterion ('het' for heterogeneity)
    seed : random seed

    Returns
    -------
    dict with keys:
        'model' : fitted CausalForestDML object
        'cate'  : array of individual treatment effects
        'cate_intervals' : (lower, upper) 95% confidence intervals
        'feature_importance' : DataFrame of feature importances
        'summary' : dict of summary statistics
    """
    from econml.dml import CausalForestDML

    # Handle NaN: fill with column median for tree-based models
    X_filled = X.fillna(X.median())

    # Nuisance models: GBM for both propensity (classification) and outcome (regression)
    propensity_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=seed,
    )
    outcome_model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=seed,
    )

    # Fit the causal forest
    cf = CausalForestDML(
        model_t=propensity_model,
        model_y=outcome_model,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        criterion=criterion,
        random_state=seed,
        cv=CAUSAL_FOREST_CV_FOLDS,
    )

    print("  [causal_forest] Fitting CausalForestDML...")
    print(f"    Trees: {n_estimators}, Min leaf: {min_samples_leaf}")
    print(f"    Features: {X_filled.shape[1]}, Patients: {X_filled.shape[0]}")

    cf.fit(
        Y=Y.values,
        T=T.values,
        X=X_filled.values,
    )

    # ── Extract CATE estimates ───────────────────────────────────────────
    cate = cf.effect(X_filled.values).flatten()
    cate_lower, cate_upper = cf.effect_interval(X_filled.values, alpha=0.05)
    cate_lower = cate_lower.flatten()
    cate_upper = cate_upper.flatten()

    # ── Feature importance ───────────────────────────────────────────────
    importance = cf.feature_importances_
    feature_imp = pd.DataFrame({
        "feature": X.columns,
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # ── Summary statistics ───────────────────────────────────────────────
    ate = cate.mean()
    sig_positive = (cate_lower > 0).sum()  # continuation harmful
    sig_negative = (cate_upper < 0).sum()  # continuation beneficial
    sig_any = sig_positive + sig_negative

    summary = {
        "ate_mean": ate,
        "ate_std": cate.std(),
        "cate_median": np.median(cate),
        "cate_iqr_25": np.percentile(cate, 25),
        "cate_iqr_75": np.percentile(cate, 75),
        "n_patients": len(cate),
        "n_sig_positive": int(sig_positive),
        "n_sig_negative": int(sig_negative),
        "n_sig_any": int(sig_any),
        "pct_sig_any": round(sig_any / len(cate) * 100, 1),
    }

    print(f"\n  [causal_forest] Results:")
    print(f"    ATE (mean CATE): {ate:.4f}")
    print(f"    CATE range: [{cate.min():.4f}, {cate.max():.4f}]")
    print(f"    Patients with significant effect: {sig_any} ({summary['pct_sig_any']}%)")
    print(f"      Continuation beneficial: {sig_negative}")
    print(f"      Continuation harmful:    {sig_positive}")

    return {
        "model": cf,
        "cate": cate,
        "cate_intervals": (cate_lower, cate_upper),
        "feature_importance": feature_imp,
        "summary": summary,
    }


def calibration_test(
    X: pd.DataFrame,
    T: pd.Series,
    Y: pd.Series,
    cate: np.ndarray,
    n_groups: int = 5,
) -> pd.DataFrame:
    """
    Calibration test: do patients with higher predicted CATE actually
    show larger observed treatment effects?

    Splits patients into quantile groups by predicted CATE, then
    computes the observed treatment effect (difference in mean Y)
    within each group.

    A well-calibrated model shows a monotonic relationship between
    predicted and observed effects.

    Parameters
    ----------
    X : Feature matrix
    T : Treatment indicator
    Y : Outcome
    cate : Individual treatment effect estimates
    n_groups : number of quantile groups

    Returns
    -------
    DataFrame with columns: group, n, predicted_effect, observed_effect
    """
    df = pd.DataFrame({
        "cate": cate,
        "T": T.values,
        "Y": Y.values,
    })

    # Create quantile groups
    df["group"] = pd.qcut(df["cate"], q=n_groups, labels=False, duplicates="drop")

    rows = []
    for g, grp in df.groupby("group"):
        treated = grp[grp["T"] == 1]["Y"]
        control = grp[grp["T"] == 0]["Y"]
        observed_effect = treated.mean() - control.mean() if len(treated) > 0 and len(control) > 0 else np.nan
        rows.append({
            "group": int(g),
            "n": len(grp),
            "n_treated": len(treated),
            "n_control": len(control),
            "predicted_effect": grp["cate"].mean(),
            "observed_effect": observed_effect,
        })

    cal_df = pd.DataFrame(rows)
    return cal_df


def subgroup_cate_summary(
    X: pd.DataFrame,
    cate: np.ndarray,
    cate_intervals: tuple[np.ndarray, np.ndarray],
    cohort_df: pd.DataFrame,
    subgroup_vars: list[tuple[str, list[str]]] | None = None,
) -> pd.DataFrame:
    """
    Compute mean CATE within pre-specified clinical subgroups.

    Parameters
    ----------
    X : Feature matrix (used for alignment)
    cate : Individual treatment effect estimates
    cate_intervals : (lower, upper) 95% CI arrays
    cohort_df : Original cohort DataFrame (for subgroup labels)
    subgroup_vars : list of (column_name, [level1, level2, ...]) tuples.
                    If None, uses a default clinical set.

    Returns
    -------
    DataFrame with one row per subgroup level showing mean CATE and CI
    """
    if subgroup_vars is None:
        subgroup_vars = [
            ("gender", None),
            ("ecog_ps", None),
            ("pdl1_status", None),
            ("histology", None),
            ("brain_mets", None),
            ("pembro_with_chemo", None),
            ("de_novo_vs_recurrent", None),
            ("smoking_history", None),
        ]

    cate_lower, cate_upper = cate_intervals
    rows = []

    for var_name, levels in subgroup_vars:
        if var_name not in cohort_df.columns:
            continue

        values = cohort_df[var_name].values
        if levels is None:
            levels = [v for v in pd.Series(values).dropna().unique()
                      if str(v).lower() not in ("unknown", "nan", "none", "")]

        for level in levels:
            mask = values == level
            if mask.sum() < 10:
                continue

            rows.append({
                "variable": var_name,
                "level": str(level),
                "n": int(mask.sum()),
                "mean_cate": float(cate[mask].mean()),
                "median_cate": float(np.median(cate[mask])),
                "ci_lower": float(cate_lower[mask].mean()),
                "ci_upper": float(cate_upper[mask].mean()),
                "pct_sig_beneficial": float(
                    (cate_upper[mask] < 0).sum() / mask.sum() * 100
                ),
                "pct_sig_harmful": float(
                    (cate_lower[mask] > 0).sum() / mask.sum() * 100
                ),
            })

    return pd.DataFrame(rows)


def run_imputation_sensitivity(
    imputed_datasets: list[pd.DataFrame],
    T: pd.Series,
    Y: pd.Series,
    **kwargs,
) -> dict:
    """
    Run the causal forest on each imputed dataset and pool results
    using Rubin's rules.

    Parameters
    ----------
    imputed_datasets : list of complete feature matrices from data_prep
    T : Treatment indicator
    Y : Outcome

    Returns
    -------
    dict with pooled ATE, within-imputation variance, between-imputation
    variance, total variance, and per-imputation results
    """
    per_imp_ate = []
    per_imp_var = []
    per_imp_cate = []

    for i, X_imp in enumerate(imputed_datasets):
        print(f"\n  [sensitivity] Imputation {i + 1}/{len(imputed_datasets)}")
        result = fit_causal_forest(X_imp, T, Y, **kwargs)
        ate = result["summary"]["ate_mean"]
        ate_var = result["summary"]["ate_std"] ** 2 / result["summary"]["n_patients"]
        per_imp_ate.append(ate)
        per_imp_var.append(ate_var)
        per_imp_cate.append(result["cate"])

    # Rubin's rules
    m = len(imputed_datasets)
    pooled_ate = np.mean(per_imp_ate)
    within_var = np.mean(per_imp_var)
    between_var = np.var(per_imp_ate, ddof=1)
    total_var = within_var + (1 + 1 / m) * between_var
    pooled_se = np.sqrt(total_var)

    # 95% CI (using normal approximation)
    pooled_ci_lower = pooled_ate - 1.96 * pooled_se
    pooled_ci_upper = pooled_ate + 1.96 * pooled_se

    # Pool individual CATEs (average across imputations)
    pooled_cate = np.mean(per_imp_cate, axis=0)

    print(f"\n  [sensitivity] Pooled ATE: {pooled_ate:.4f} "
          f"(95% CI: {pooled_ci_lower:.4f} to {pooled_ci_upper:.4f})")
    print(f"    Within-imputation variance: {within_var:.6f}")
    print(f"    Between-imputation variance: {between_var:.6f}")

    return {
        "pooled_ate": pooled_ate,
        "pooled_se": pooled_se,
        "pooled_ci": (pooled_ci_lower, pooled_ci_upper),
        "within_var": within_var,
        "between_var": between_var,
        "total_var": total_var,
        "per_imputation_ate": per_imp_ate,
        "pooled_cate": pooled_cate,
    }
