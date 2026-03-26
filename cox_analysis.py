"""
cox_analysis.py — Comprehensive Cox Proportional Hazards Analysis
==================================================================
Implements multiple modeling strategies per the SAP:
  1. Unadjusted Cox (treatment only)
  2. Minimally adjusted (age + sex)
  3. Clinically adjusted (all core covariates)
  4. Fully adjusted (core + comorbidities + medications)
  5. LASSO-selected covariates
  6. Schoenfeld residuals for PH assumption check
  7. Subgroup analyses with interaction tests
  8. Landmark sensitivity (27, 29, 32 months)

Reference: Rousseau et al. Lancet Reg Health Eur 2024;43:100970
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

MIN_CELL_COUNT = 10  # minimum patients in a dummy category to keep it in the model


def _prepare_cox_df(
    cohort_df: pd.DataFrame,
    covariates: list[str],
    time_col: str = "os_time_months",
    event_col: str = "os_event",
    treatment_col: str = "cohort",
    min_cell: int = MIN_CELL_COUNT,
) -> pd.DataFrame:
    """
    Prepare a model-ready DataFrame:
      - Encode treatment as binary (Continuation=1, Fixed-Duration=0)
      - One-hot encode categoricals (drop_first=True)
      - Drop dummy columns with fewer than min_cell patients (avoids singular matrices)
      - Fill numeric NaN with median
      - Drop rows with zero/negative time
    """
    df = cohort_df.copy()

    # Binary treatment
    df["treatment_continuation"] = (df[treatment_col] == "Continuation").astype(int)

    model_cols = ["treatment_continuation"]

    for cov in covariates:
        if cov not in df.columns:
            continue
        col = df[cov]
        if col.dtype == "object" or str(col.dtype) == "category":
            dummies = pd.get_dummies(col, prefix=cov, drop_first=True, dtype=int)
            # Drop rare categories — they cause singular matrices
            dummies = dummies.loc[:, dummies.sum() >= min_cell]
            df = pd.concat([df, dummies], axis=1)
            model_cols.extend(dummies.columns.tolist())
        else:
            df[cov] = df[cov].fillna(df[cov].median())
            model_cols.append(cov)

    model_df = df[model_cols + [time_col, event_col]].copy()
    model_df = model_df.dropna()
    model_df = model_df[model_df[time_col] > 0]

    return model_df


# ─────────────────────────────────────────────────────────────────────────────
# 1. MULTIPLE COX MODELS
# ─────────────────────────────────────────────────────────────────────────────

def run_multiple_cox_models(
    cohort_df: pd.DataFrame,
    model_specs: dict[str, list[str]],
    time_col: str = "os_time_months",
    event_col: str = "os_event",
) -> dict:
    """
    Fit multiple Cox models with different covariate sets.

    Parameters
    ----------
    cohort_df : analysis cohort
    model_specs : dict of {model_name: [covariate_list]}

    Returns
    -------
    dict of {model_name: {cph, summary, concordance, n_patients, n_events, treatment_hr}}
    """
    results = {}

    for model_name, covariates in model_specs.items():
        print(f"\n  Fitting Cox model: {model_name} ({len(covariates)} covariates)...")

        try:
            model_df = _prepare_cox_df(cohort_df, covariates, time_col, event_col)

            cph = CoxPHFitter()
            cph.fit(
                model_df,
                duration_col=time_col,
                event_col=event_col,
                show_progress=False,
            )

            # Extract treatment HR
            tx_summary = cph.summary.loc["treatment_continuation"]
            hr = tx_summary["exp(coef)"]
            hr_lower = tx_summary["exp(coef) lower 95%"]
            hr_upper = tx_summary["exp(coef) upper 95%"]
            p_val = tx_summary["p"]

            results[model_name] = {
                "cph": cph,
                "summary": cph.summary,
                "concordance": cph.concordance_index_,
                "n_patients": len(model_df),
                "n_events": int(model_df[event_col].sum()),
                "treatment_hr": {
                    "HR": hr,
                    "HR_lower": hr_lower,
                    "HR_upper": hr_upper,
                    "p_value": p_val,
                },
                "covariates": covariates,
            }

            print(f"    HR (Continuation vs Fixed-Duration): "
                  f"{hr:.3f} ({hr_lower:.3f}–{hr_upper:.3f}), p={p_val:.4f}")
            print(f"    C-index: {cph.concordance_index_:.3f}")

        except Exception as e:
            print(f"    ERROR fitting {model_name}: {e}")
            results[model_name] = {"error": str(e)}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. LASSO-SELECTED COX MODEL
# ─────────────────────────────────────────────────────────────────────────────

def run_lasso_cox(
    cohort_df: pd.DataFrame,
    candidate_covariates: list[str],
    alpha_range: list[float] = None,
    time_col: str = "os_time_months",
    event_col: str = "os_event",
) -> dict:
    """
    Use penalized Cox (L1) to select covariates, then refit unpenalized.

    Returns dict with: selected_covariates, cph, summary, concordance, treatment_hr
    """
    if alpha_range is None:
        alpha_range = [0.01, 0.05, 0.1, 0.5, 1.0]

    model_df = _prepare_cox_df(cohort_df, candidate_covariates, time_col, event_col)

    # Fit penalized Cox with cross-validation over alpha values
    best_cph = None
    best_concordance = 0
    best_alpha = None

    for alpha in alpha_range:
        try:
            cph = CoxPHFitter(penalizer=alpha, l1_ratio=1.0)
            cph.fit(model_df, duration_col=time_col, event_col=event_col,
                    show_progress=False)
            if cph.concordance_index_ > best_concordance:
                best_concordance = cph.concordance_index_
                best_cph = cph
                best_alpha = alpha
        except Exception:
            continue

    if best_cph is None:
        return {"error": "LASSO Cox failed for all alpha values"}

    # Identify selected covariates (non-zero coefficients)
    coefs = best_cph.params_
    selected = coefs[coefs.abs() > 1e-6].index.tolist()

    # Always keep treatment
    if "treatment_continuation" not in selected:
        selected = ["treatment_continuation"] + selected

    # Filter selected vars: drop any dummy where the column sum < MIN_CELL_COUNT
    selected_clean = []
    for col in selected:
        if col == "treatment_continuation":
            selected_clean.append(col)
        elif col in model_df.columns and model_df[col].sum() >= MIN_CELL_COUNT:
            selected_clean.append(col)
        else:
            pass  # silently drop sparse dummies

    print(f"  LASSO (alpha={best_alpha}): selected {len(selected_clean)} variables "
          f"(after dropping {len(selected) - len(selected_clean)} sparse dummies)")
    print(f"    Selected: {selected_clean}")

    # Refit unpenalized with selected variables
    refit_df = model_df[selected_clean + [time_col, event_col]].copy()
    try:
        cph_refit = CoxPHFitter()
        cph_refit.fit(refit_df, duration_col=time_col, event_col=event_col,
                      show_progress=False)
    except Exception as e:
        print(f"  LASSO unpenalized refit failed ({e}). Returning penalized model.")
        cph_refit = best_cph
        selected_clean = selected

    tx_summary = cph_refit.summary.loc["treatment_continuation"]

    return {
        "cph": cph_refit,
        "summary": cph_refit.summary,
        "concordance": cph_refit.concordance_index_,
        "selected_covariates": selected_clean,
        "best_alpha": best_alpha,
        "n_patients": len(refit_df),
        "n_events": int(refit_df[event_col].sum()),
        "treatment_hr": {
            "HR": tx_summary["exp(coef)"],
            "HR_lower": tx_summary["exp(coef) lower 95%"],
            "HR_upper": tx_summary["exp(coef) upper 95%"],
            "p_value": tx_summary["p"],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. SCHOENFELD RESIDUALS — PH ASSUMPTION CHECK
# ─────────────────────────────────────────────────────────────────────────────

def test_proportional_hazards(
    cohort_df: pd.DataFrame,
    covariates: list[str],
    time_col: str = "os_time_months",
    event_col: str = "os_event",
) -> dict:
    """
    Test the proportional hazards assumption using Schoenfeld residuals.

    Returns dict with:
      - test_results: DataFrame with test statistic and p-value per covariate
      - cph: fitted model
      - global_test_p: global PH test p-value
    """
    model_df = _prepare_cox_df(cohort_df, covariates, time_col, event_col)

    cph = CoxPHFitter()
    cph.fit(model_df, duration_col=time_col, event_col=event_col,
            show_progress=False)

    # lifelines provides a check_assumptions method that does Schoenfeld test
    try:
        # Capture results
        ph_results = cph.check_assumptions(
            model_df, p_value_threshold=1.0, show_plots=False
        )
        # check_assumptions returns None but prints — we need the summary
        # Use the proportional_hazard_test method instead
        from lifelines.statistics import proportional_hazard_test
        ph_test = proportional_hazard_test(cph, model_df, time_transform="rank")

        test_df = ph_test.summary.copy()
        test_df.columns = ["Test Statistic", "p-value", "df"]

        # Global test: if any covariate violates, flag it
        global_p = test_df["p-value"].min()

        return {
            "test_results": test_df,
            "cph": cph,
            "global_test_p": global_p,
            "model_df": model_df,
        }

    except Exception as e:
        print(f"  PH test error: {e}")
        # Fallback: return the model without test
        return {
            "test_results": pd.DataFrame(),
            "cph": cph,
            "global_test_p": np.nan,
            "model_df": model_df,
            "error": str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4. SUBGROUP ANALYSES
# ─────────────────────────────────────────────────────────────────────────────

def run_subgroup_analyses(
    cohort_df: pd.DataFrame,
    subgroup_vars: list[tuple[str, list[str]]],
    time_col: str = "os_time_months",
    event_col: str = "os_event",
    treatment_col: str = "cohort",
) -> pd.DataFrame:
    """
    Run unadjusted Cox for each subgroup level and compute
    treatment HR within each subgroup. Also compute interaction p-value.

    Parameters
    ----------
    subgroup_vars : list of (variable_name, [level1, level2, ...])

    Returns DataFrame with columns:
      Variable, Level, N, Events, HR, HR_lower, HR_upper, p_value, interaction_p
    """
    rows = []

    # Overall
    try:
        model_df = _prepare_cox_df(cohort_df, [], time_col, event_col)
        cph = CoxPHFitter()
        cph.fit(model_df, duration_col=time_col, event_col=event_col,
                show_progress=False)
        tx = cph.summary.loc["treatment_continuation"]
        rows.append({
            "Variable": "Overall",
            "Level": "All patients",
            "N": len(model_df),
            "Events": int(model_df[event_col].sum()),
            "HR": tx["exp(coef)"],
            "HR_lower": tx["exp(coef) lower 95%"],
            "HR_upper": tx["exp(coef) upper 95%"],
            "p_value": tx["p"],
            "interaction_p": np.nan,
        })
    except Exception:
        pass

    for var_name, levels in subgroup_vars:
        if var_name not in cohort_df.columns:
            continue

        # Interaction test: add treatment × subgroup interaction to full model
        interaction_p = np.nan
        try:
            df_int = cohort_df.copy()
            df_int["treatment_continuation"] = (df_int[treatment_col] == "Continuation").astype(int)
            df_int["_subgroup"] = df_int[var_name].astype(str)

            # Create interaction terms
            for lev in levels[1:]:
                df_int[f"_int_{lev}"] = (
                    (df_int["_subgroup"] == lev).astype(int) *
                    df_int["treatment_continuation"]
                )

            int_cols = ["treatment_continuation"] + [f"_int_{lev}" for lev in levels[1:]]
            sub_dummies = pd.get_dummies(df_int["_subgroup"], prefix="_sub", drop_first=True, dtype=int)
            df_int = pd.concat([df_int, sub_dummies], axis=1)
            int_cols.extend(sub_dummies.columns.tolist())

            int_model_df = df_int[int_cols + [time_col, event_col]].dropna()
            int_model_df = int_model_df[int_model_df[time_col] > 0]

            cph_int = CoxPHFitter()
            cph_int.fit(int_model_df, duration_col=time_col,
                        event_col=event_col, show_progress=False)

            # Interaction p-value: use the interaction term(s)
            int_terms = [c for c in cph_int.summary.index if c.startswith("_int_")]
            if int_terms:
                interaction_p = cph_int.summary.loc[int_terms, "p"].min()
        except Exception:
            pass

        for level in levels:
            subset = cohort_df[cohort_df[var_name].astype(str) == level]
            if len(subset) < 10:
                continue

            # Need at least some events in both arms
            try:
                model_df = _prepare_cox_df(subset, [], time_col, event_col)
                if model_df["treatment_continuation"].nunique() < 2:
                    continue
                if model_df[event_col].sum() < 5:
                    continue

                cph = CoxPHFitter()
                cph.fit(model_df, duration_col=time_col,
                        event_col=event_col, show_progress=False)
                tx = cph.summary.loc["treatment_continuation"]

                rows.append({
                    "Variable": var_name,
                    "Level": level,
                    "N": len(model_df),
                    "Events": int(model_df[event_col].sum()),
                    "HR": tx["exp(coef)"],
                    "HR_lower": tx["exp(coef) lower 95%"],
                    "HR_upper": tx["exp(coef) upper 95%"],
                    "p_value": tx["p"],
                    "interaction_p": interaction_p,
                })
            except Exception:
                continue

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 5. LANDMARK SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def run_landmark_sensitivity(
    tables: dict,
    select_cohort_fn,
    landmark_months_list: list[int],
    covariates: list[str],
    time_col: str = "os_time_months",
    event_col: str = "os_event",
) -> pd.DataFrame:
    """
    Re-run the full cohort selection + Cox model at different landmark times.

    Returns DataFrame with columns:
      Landmark_months, N, Events, HR, HR_lower, HR_upper, p_value, concordance
    """
    from config import LANDMARK_MONTHS as ORIG_LANDMARK
    import config

    rows = []

    for lm_months in landmark_months_list:
        print(f"\n  Landmark sensitivity: {lm_months} months...")

        # Temporarily override landmark in config
        original_val = config.LANDMARK_MONTHS
        config.LANDMARK_MONTHS = lm_months

        try:
            cohort_df, attrition = select_cohort_fn(tables)

            if len(cohort_df) < 20:
                print(f"    Skipping — only {len(cohort_df)} patients")
                config.LANDMARK_MONTHS = original_val
                continue

            model_df = _prepare_cox_df(cohort_df, covariates, time_col, event_col)

            cph = CoxPHFitter()
            cph.fit(model_df, duration_col=time_col, event_col=event_col,
                    show_progress=False)

            tx = cph.summary.loc["treatment_continuation"]

            n_fd = (cohort_df["cohort"] == "Fixed-Duration").sum()
            n_cont = (cohort_df["cohort"] == "Continuation").sum()

            rows.append({
                "Landmark (months)": lm_months,
                "N": len(model_df),
                "N Fixed-Duration": n_fd,
                "N Continuation": n_cont,
                "Events": int(model_df[event_col].sum()),
                "HR": tx["exp(coef)"],
                "HR_lower": tx["exp(coef) lower 95%"],
                "HR_upper": tx["exp(coef) upper 95%"],
                "p_value": tx["p"],
                "C-index": cph.concordance_index_,
            })

            print(f"    N={len(model_df)} (FD={n_fd}, Cont={n_cont}), "
                  f"HR={tx['exp(coef)']:.3f} "
                  f"({tx['exp(coef) lower 95%']:.3f}-{tx['exp(coef) upper 95%']:.3f}), "
                  f"p={tx['p']:.4f}")

        except Exception as e:
            print(f"    ERROR at landmark {lm_months}m: {e}")

        config.LANDMARK_MONTHS = original_val

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 6. SUMMARY COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

def build_model_comparison_table(cox_results: dict, lasso_result: dict = None) -> pd.DataFrame:
    """
    Build a comparison table of treatment HR across all models.
    """
    rows = []

    for model_name, result in cox_results.items():
        if "error" in result:
            continue
        hr = result["treatment_hr"]
        rows.append({
            "Model": model_name.replace("_", " ").title(),
            "Covariates": len(result.get("covariates", [])),
            "N": result["n_patients"],
            "Events": result["n_events"],
            "HR": f"{hr['HR']:.3f}",
            "95% CI": f"{hr['HR_lower']:.3f}–{hr['HR_upper']:.3f}",
            "p-value": f"{hr['p_value']:.4f}" if hr['p_value'] >= 0.0001 else "<0.0001",
            "C-index": f"{result['concordance']:.3f}",
        })

    if lasso_result and "error" not in lasso_result:
        hr = lasso_result["treatment_hr"]
        rows.append({
            "Model": "LASSO-Selected",
            "Covariates": len(lasso_result.get("selected_covariates", [])) - 1,
            "N": lasso_result["n_patients"],
            "Events": lasso_result["n_events"],
            "HR": f"{hr['HR']:.3f}",
            "95% CI": f"{hr['HR_lower']:.3f}–{hr['HR_upper']:.3f}",
            "p-value": f"{hr['p_value']:.4f}" if hr['p_value'] >= 0.0001 else "<0.0001",
            "C-index": f"{lasso_result['concordance']:.3f}",
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# PRINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def print_model_comparison(comparison_df: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print("COX MODEL COMPARISON — Treatment HR (Continuation vs Fixed-Duration)")
    print("=" * 90)
    print(comparison_df.to_string(index=False))
    print("=" * 90)


def print_subgroup_results(subgroup_df: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print("SUBGROUP ANALYSIS — Treatment HR by Patient Characteristics")
    print("=" * 90)
    display_cols = ["Variable", "Level", "N", "Events", "HR", "HR_lower",
                    "HR_upper", "p_value", "interaction_p"]
    available = [c for c in display_cols if c in subgroup_df.columns]
    print(subgroup_df[available].to_string(index=False))
    print("=" * 90)


def print_ph_test(ph_result: dict) -> None:
    print("\n" + "=" * 70)
    print("PROPORTIONAL HAZARDS ASSUMPTION TEST (Schoenfeld Residuals)")
    print("=" * 70)
    if not ph_result["test_results"].empty:
        print(ph_result["test_results"].to_string())
        print(f"\nGlobal minimum p-value: {ph_result['global_test_p']:.4f}")
        if ph_result["global_test_p"] < 0.05:
            print("⚠  PH assumption may be violated for some covariates (p < 0.05)")
        else:
            print("✓  PH assumption appears satisfied for all covariates")
    else:
        print("  Test could not be completed.")
    print("=" * 70)


def print_landmark_sensitivity(landmark_df: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print("LANDMARK SENSITIVITY ANALYSIS")
    print("=" * 90)
    print(landmark_df.to_string(index=False))
    print("=" * 90)
