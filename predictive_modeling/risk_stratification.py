"""
risk_stratification.py — Risk Stratification and Conditional Treatment Effects
================================================================================
Builds a prognostic risk model (overall mortality risk regardless of treatment),
stratifies patients into risk groups, and then examines whether continuation
pembrolizumab provides differential benefit across risk strata.

Clinical rationale:
  If high-risk patients (poor ECOG, brain mets, unfavorable histology) benefit
  more from continuing pembrolizumab, that's an actionable finding — even if
  the average treatment effect is null.

Approach:
  1. Fit a survival model (Random Survival Forest or Cox with elastic net)
     on baseline features to predict OS risk
  2. Stratify patients into risk tertiles (low / medium / high)
  3. Within each stratum, compare OS between fixed-duration and continuation
     using Kaplan-Meier and Cox regression
  4. Test for interaction between risk group and treatment
"""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold

from config import (
    RISK_N_GROUPS,
    RISK_MODEL_N_ESTIMATORS,
    RISK_MODEL_SEED,
    OUTCOME_EVENT_COL,
    OUTCOME_TIME_COL,
    TREATMENT_COL,
    TREATMENT_POSITIVE,
    TREATMENT_NEGATIVE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_risk_model(
    X: pd.DataFrame,
    Y_event: pd.Series,
    Y_time: pd.Series,
    n_groups: int = RISK_N_GROUPS,
    n_estimators: int = RISK_MODEL_N_ESTIMATORS,
    seed: int = RISK_MODEL_SEED,
) -> dict:
    """
    Build a prognostic risk model and stratify patients.

    Uses a Random Forest classifier trained to predict the binary event
    (death within follow-up), with cross-validated predicted probabilities
    to avoid overfitting.

    Parameters
    ----------
    X : Feature matrix
    Y_event : Binary event indicator
    Y_time : Survival time in months
    n_groups : Number of risk strata
    n_estimators : Trees in the random forest
    seed : Random state

    Returns
    -------
    dict with:
        'risk_scores' : array of predicted mortality probabilities
        'risk_groups' : array of risk group labels ('Low', 'Medium', 'High')
        'risk_group_labels' : ordered list of group names
        'model' : fitted RandomForest (on full data, for feature importance)
        'feature_importance' : DataFrame of feature importances
        'group_summary' : DataFrame summarizing each risk group
    """
    X_filled = X.fillna(X.median())

    # Cross-validated risk scores to avoid overfitting
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=8,
        min_samples_leaf=20,
        random_state=seed,
        n_jobs=-1,
    )

    print("  [risk_model] Computing cross-validated risk scores...")
    risk_scores = cross_val_predict(
        rf, X_filled, Y_event, cv=cv, method="predict_proba"
    )[:, 1]  # probability of death

    # Fit final model on full data for feature importance
    rf.fit(X_filled, Y_event)
    feature_imp = pd.DataFrame({
        "feature": X.columns,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # Stratify into risk groups using quantile cuts
    group_labels = _make_group_labels(n_groups)
    risk_groups = pd.qcut(
        risk_scores, q=n_groups, labels=group_labels, duplicates="drop"
    ).astype(str)

    # Summary per group
    group_summary = _summarize_risk_groups(
        risk_scores, risk_groups, Y_event, Y_time, group_labels
    )

    print(f"\n  [risk_model] Risk stratification ({n_groups} groups):")
    print(group_summary.to_string(index=False))

    return {
        "risk_scores": risk_scores,
        "risk_groups": np.array(risk_groups),
        "risk_group_labels": group_labels,
        "model": rf,
        "feature_importance": feature_imp,
        "group_summary": group_summary,
    }


def treatment_effect_by_risk_group(
    cohort_df: pd.DataFrame,
    risk_groups: np.ndarray,
    group_labels: list[str],
) -> pd.DataFrame:
    """
    Within each risk stratum, compare OS between treatment arms.

    For each group:
      - Kaplan-Meier median OS per arm
      - Log-rank p-value
      - Cox HR with 95% CI

    Parameters
    ----------
    cohort_df : analysis cohort with treatment and survival columns
    risk_groups : array of group labels aligned with cohort_df
    group_labels : ordered list of group names

    Returns
    -------
    DataFrame with one row per risk group
    """
    rows = []
    for label in group_labels:
        mask = risk_groups == label
        sub = cohort_df[mask].copy()
        n = len(sub)

        cont = sub[sub[TREATMENT_COL] == TREATMENT_POSITIVE]
        fixed = sub[sub[TREATMENT_COL] == TREATMENT_NEGATIVE]

        if len(cont) < 5 or len(fixed) < 5:
            rows.append({
                "risk_group": label,
                "n": n,
                "n_continuation": len(cont),
                "n_fixed_duration": len(fixed),
                "median_os_continuation": np.nan,
                "median_os_fixed_duration": np.nan,
                "log_rank_p": np.nan,
                "cox_hr": np.nan,
                "cox_hr_lower": np.nan,
                "cox_hr_upper": np.nan,
                "cox_p": np.nan,
            })
            continue

        # Kaplan-Meier
        kmf_cont = KaplanMeierFitter()
        kmf_cont.fit(cont[OUTCOME_TIME_COL], cont[OUTCOME_EVENT_COL])
        kmf_fixed = KaplanMeierFitter()
        kmf_fixed.fit(fixed[OUTCOME_TIME_COL], fixed[OUTCOME_EVENT_COL])

        # Log-rank test
        lr = logrank_test(
            cont[OUTCOME_TIME_COL], fixed[OUTCOME_TIME_COL],
            cont[OUTCOME_EVENT_COL], fixed[OUTCOME_EVENT_COL],
        )

        # Cox PH for hazard ratio
        cox_data = sub[[OUTCOME_TIME_COL, OUTCOME_EVENT_COL, TREATMENT_COL]].copy()
        cox_data["treatment"] = (cox_data[TREATMENT_COL] == TREATMENT_POSITIVE).astype(int)
        cox_data = cox_data[[OUTCOME_TIME_COL, OUTCOME_EVENT_COL, "treatment"]].copy()
        # Remove zero-time rows
        cox_data = cox_data[cox_data[OUTCOME_TIME_COL] > 0]

        try:
            cph = CoxPHFitter()
            cph.fit(
                cox_data,
                duration_col=OUTCOME_TIME_COL,
                event_col=OUTCOME_EVENT_COL,
            )
            hr = np.exp(cph.params_["treatment"])
            ci = np.exp(cph.confidence_intervals_.loc["treatment"])
            hr_lower = ci.iloc[0]
            hr_upper = ci.iloc[1]
            cox_p = cph.summary.loc["treatment", "p"]
        except Exception:
            hr = hr_lower = hr_upper = cox_p = np.nan

        rows.append({
            "risk_group": label,
            "n": n,
            "n_continuation": len(cont),
            "n_fixed_duration": len(fixed),
            "median_os_continuation": kmf_cont.median_survival_time_,
            "median_os_fixed_duration": kmf_fixed.median_survival_time_,
            "log_rank_p": lr.p_value,
            "cox_hr": hr,
            "cox_hr_lower": hr_lower,
            "cox_hr_upper": hr_upper,
            "cox_p": cox_p,
        })

    result = pd.DataFrame(rows)

    print("\n  [risk_strat] Treatment effect by risk group:")
    for _, row in result.iterrows():
        print(f"    {row['risk_group']:8s}  N={row['n']:4d}  "
              f"HR={row['cox_hr']:.3f} "
              f"({row['cox_hr_lower']:.3f}-{row['cox_hr_upper']:.3f})  "
              f"p={row['cox_p']:.4f}")

    return result


def interaction_test(
    cohort_df: pd.DataFrame,
    risk_groups: np.ndarray,
) -> dict:
    """
    Test for statistical interaction between risk group and treatment
    using a Cox model with interaction terms.

    Model: OS ~ treatment + risk_group + treatment * risk_group

    A significant interaction term suggests the treatment effect differs
    across risk strata.

    Parameters
    ----------
    cohort_df : analysis cohort
    risk_groups : risk group assignments

    Returns
    -------
    dict with interaction model summary
    """
    df = cohort_df[[OUTCOME_TIME_COL, OUTCOME_EVENT_COL, TREATMENT_COL]].copy()
    df["treatment"] = (df[TREATMENT_COL] == TREATMENT_POSITIVE).astype(int)
    df["risk_group"] = risk_groups

    # Encode risk groups as dummies (drop first = Low risk as reference)
    risk_dummies = pd.get_dummies(df["risk_group"], prefix="risk", drop_first=True)
    df = pd.concat([df, risk_dummies], axis=1)

    # Create interaction terms
    interaction_cols = []
    for col in risk_dummies.columns:
        int_col = f"treatment_x_{col}"
        df[int_col] = df["treatment"] * df[col]
        interaction_cols.append(int_col)

    # Fit Cox model
    cox_cols = (
        [OUTCOME_TIME_COL, OUTCOME_EVENT_COL, "treatment"]
        + list(risk_dummies.columns)
        + interaction_cols
    )
    cox_data = df[cox_cols].copy()
    cox_data = cox_data[cox_data[OUTCOME_TIME_COL] > 0]

    try:
        cph = CoxPHFitter()
        cph.fit(
            cox_data,
            duration_col=OUTCOME_TIME_COL,
            event_col=OUTCOME_EVENT_COL,
        )

        # Extract interaction term p-values
        interaction_results = {}
        for col in interaction_cols:
            if col in cph.summary.index:
                interaction_results[col] = {
                    "coef": cph.params_[col],
                    "hr": np.exp(cph.params_[col]),
                    "p": cph.summary.loc[col, "p"],
                }

        # Overall interaction test: LRT comparing model with vs without interactions
        # Simplified: use the minimum p-value among interaction terms
        min_p = min(v["p"] for v in interaction_results.values()) if interaction_results else 1.0

        print(f"\n  [interaction] Risk group x treatment interaction test:")
        for name, vals in interaction_results.items():
            print(f"    {name}: HR={vals['hr']:.3f}, p={vals['p']:.4f}")
        print(f"    Most significant interaction p-value: {min_p:.4f}")

        return {
            "interaction_terms": interaction_results,
            "model_summary": cph.summary,
            "min_interaction_p": min_p,
            "significant": min_p < 0.05,
        }

    except Exception as e:
        print(f"  [interaction] Could not fit interaction model: {e}")
        return {
            "interaction_terms": {},
            "model_summary": None,
            "min_interaction_p": np.nan,
            "significant": False,
            "error": str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_group_labels(n_groups: int) -> list[str]:
    """Generate ordered group labels."""
    if n_groups == 2:
        return ["Low Risk", "High Risk"]
    elif n_groups == 3:
        return ["Low Risk", "Medium Risk", "High Risk"]
    elif n_groups == 4:
        return ["Low Risk", "Low-Medium Risk", "Medium-High Risk", "High Risk"]
    else:
        return [f"Risk Q{i + 1}" for i in range(n_groups)]


def _summarize_risk_groups(
    risk_scores: np.ndarray,
    risk_groups: np.ndarray,
    Y_event: pd.Series,
    Y_time: pd.Series,
    group_labels: list[str],
) -> pd.DataFrame:
    """Compute summary statistics per risk group."""
    rows = []
    for label in group_labels:
        mask = risk_groups == label
        if mask.sum() == 0:
            continue
        events = Y_event[mask]
        times = Y_time[mask]
        scores = risk_scores[mask]

        rows.append({
            "risk_group": label,
            "n": int(mask.sum()),
            "event_rate": f"{events.mean():.1%}",
            "median_os_months": f"{times.median():.1f}",
            "mean_risk_score": f"{scores.mean():.3f}",
            "risk_score_range": f"{scores.min():.3f}-{scores.max():.3f}",
        })

    return pd.DataFrame(rows)
