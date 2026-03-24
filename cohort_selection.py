"""
cohort_selection.py — Implement SAP inclusion/exclusion criteria
================================================================
Produces an analysis-ready DataFrame with one row per patient including:
  - cohort assignment (fixed_duration vs continuation)
  - landmark date
  - survival time from landmark
  - event indicator
  - all covariates for Cox model

Steps follow the SAP sequentially:
  1. Metastatic NSCLC with C34.X diagnosis
  2. Diagnosed 2016-01-01 to 2025-08-31
  3. Pembrolizumab in first metastatic LOT
  4. Apply 6-month gap rule to determine effective treatment end
  5. At least one pembro infusion between 22-26 months
  6. Exclude if chemo within 2 months of last pembro infusion
  7. Assign cohort: fixed-duration (last infusion 22-26 mo) vs continuation (>26 mo)
  8. Alive at 29-month landmark
  9. Derive covariates and survival outcomes
"""

import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from config import (
    DIAGNOSIS_START, DIAGNOSIS_END, FOLLOWUP_CUTOFF,
    PEMBRO_GENERIC_NAMES, PEMBRO_BRAND_NAMES,
    FIXED_DURATION_LOWER_MONTHS, FIXED_DURATION_UPPER_MONTHS,
    CONTINUATION_LOWER_MONTHS, MAX_INFUSION_GAP_DAYS,
    LANDMARK_MONTHS, CHEMO_PROXIMITY_EXCLUSION_DAYS,
    CHEMO_AGENTS,
)


def _months_between(start: pd.Series, end: pd.Series) -> pd.Series:
    """Approximate months between two date series."""
    return (end - start).dt.days / 30.44


def _apply_gap_rule(dose_df: pd.DataFrame, max_gap_days: int) -> pd.DataFrame:
    """
    For each patient, if consecutive pembro infusions are >max_gap_days apart,
    treatment is considered to have ended at the earlier date.

    Returns DataFrame with columns: mpi_id, effective_last_infusion, all_infusion_dates
    """
    pembro_mask = (
        dose_df["generic_name"].str.lower().str.contains("pembrolizumab", na=False) |
        dose_df["brand_name"].str.lower().str.contains("keytruda", na=False)
    )
    pembro_doses = dose_df[pembro_mask].copy()

    # Diagnostic: show what was found (or what names exist if nothing matched)
    print(f"  [gap_rule] Pembro dose rows found: {len(pembro_doses):,}")
    if pembro_doses.empty:
        print("  [gap_rule] WARNING: No pembrolizumab rows matched.")
        print("  [gap_rule] Sample generic_name values in dose table:")
        print(dose_df["generic_name"].dropna().str.lower().value_counts().head(10).to_string())
        return pd.DataFrame(columns=["mpi_id", "effective_last_infusion", "first_infusion", "n_infusions"])

    pembro_doses = pembro_doses.sort_values(["mpi_id", "drug_exposure_start_date"])

    results = []
    for mpi_id, grp in pembro_doses.groupby("mpi_id"):
        dates = grp["drug_exposure_start_date"].dropna().sort_values().reset_index(drop=True)
        if len(dates) == 0:
            continue

        effective_end = dates.iloc[-1]  # default: last infusion
        for j in range(1, len(dates)):
            gap = (dates.iloc[j] - dates.iloc[j - 1]).days
            if gap > max_gap_days:
                effective_end = dates.iloc[j - 1]
                break

        results.append({
            "mpi_id": mpi_id,
            "effective_last_infusion": effective_end,
            "first_infusion": dates.iloc[0],
            "n_infusions": len(dates),
        })

    return pd.DataFrame(results)


def select_cohort(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """
    Apply SAP inclusion/exclusion criteria.

    Parameters
    ----------
    tables : dict of DataFrames from data_loader.load_tables()

    Returns
    -------
    cohort_df : analysis-ready DataFrame (one row per patient)
    attrition : dict tracking patient counts at each step
    """
    demo = tables["demographics"].copy()
    disease = tables["disease"].copy()
    lot = tables["lot"].copy()
    dose = tables["dose"].copy()
    biomarker = tables.get("biomarker", pd.DataFrame())
    labs = tables.get("labs", pd.DataFrame())
    riskscores = tables.get("riskscores", pd.DataFrame())   # fallback for ECOG
    metastases = tables.get("metastases", pd.DataFrame())

    # ── Diagnostic: print column names for key tables ───────────────────
    for tname, tdf in [("demographics", demo), ("disease", disease), ("lot", lot), ("dose", dose)]:
        print(f"  [DEBUG] {tname} columns: {list(tdf.columns[:8])}")

    attrition = {}
    attrition["total_patients"] = demo["mpi_id"].nunique()

    # ── Step 1: Metastatic NSCLC with C34.X ─────────────────────────────
    disease_met = disease[
        disease["cancer_code"].str.upper().str.startswith("C34", na=False) &
        disease["metastatic_date"].notna()
    ].copy()
    eligible_ids = set(disease_met["mpi_id"].unique())
    attrition["metastatic_nsclc_c34x"] = len(eligible_ids)

    # ── Step 2: Diagnosis date window ───────────────────────────────────
    disease_met = disease_met[
        (disease_met["diag_date"] >= DIAGNOSIS_START) &
        (disease_met["diag_date"] <= DIAGNOSIS_END)
    ]
    eligible_ids &= set(disease_met["mpi_id"].unique())
    attrition["diagnosis_window"] = len(eligible_ids)

    # ── Step 3: Pembrolizumab in first metastatic LOT ───────────────────
    lot1_met = lot[
        (lot["lot"] == 1) &
        (lot["metastatic"] == 1) &
        (lot["mpi_id"].isin(eligible_ids))
    ].copy()

    # Check that regimen contains pembrolizumab
    pembro_lot1 = lot1_met[
        lot1_met["regimen"].str.lower().str.contains("pembrolizumab", na=False)
    ]
    eligible_ids &= set(pembro_lot1["mpi_id"].unique())
    attrition["pembro_1l_metastatic"] = len(eligible_ids)

    # ── Step 4: Apply 6-month gap rule to determine effective tx end ────
    # Filter dose records to eligible patients
    dose_eligible = dose[dose["mpi_id"].isin(eligible_ids)].copy()
    infusion_summary = _apply_gap_rule(dose_eligible, MAX_INFUSION_GAP_DAYS)
    eligible_ids &= set(infusion_summary["mpi_id"].unique())

    # Merge LOT start date
    lot_starts = pembro_lot1[["mpi_id", "start_date", "regimen"]].drop_duplicates("mpi_id")
    infusion_summary = infusion_summary.merge(lot_starts, on="mpi_id", how="inner")

    # Calculate months from LOT start to effective last infusion
    infusion_summary["months_on_tx"] = _months_between(
        infusion_summary["start_date"],
        infusion_summary["effective_last_infusion"],
    )
    attrition["has_infusion_data"] = len(eligible_ids)

    # ── Step 5: At least one infusion between 22-26 months ──────────────
    # This means effective_last_infusion >= 22 months (they were still on at 22 mo)
    infusion_summary_eligible = infusion_summary[
        infusion_summary["months_on_tx"] >= FIXED_DURATION_LOWER_MONTHS
    ]
    eligible_ids &= set(infusion_summary_eligible["mpi_id"].unique())
    attrition["infusion_at_22_26_months"] = len(eligible_ids)

    # ── Step 6: Exclude if chemo within 2 months of last pembro ─────────
    # (indicates progression-motivated stop)
    chemo_mask = dose_eligible["generic_name"].str.lower().isin(CHEMO_AGENTS)
    chemo_doses = dose_eligible[chemo_mask][["mpi_id", "drug_exposure_start_date"]].copy()
    chemo_doses.rename(columns={"drug_exposure_start_date": "chemo_date"}, inplace=True)

    if not chemo_doses.empty:
        # For each patient, find last chemo date
        last_chemo = chemo_doses.groupby("mpi_id")["chemo_date"].max().reset_index()
        last_chemo.rename(columns={"chemo_date": "last_chemo_date"}, inplace=True)

        check_df = infusion_summary_eligible.merge(last_chemo, on="mpi_id", how="left")
        # Exclude if chemo within 60 days of effective last pembro infusion
        check_df["chemo_near_end"] = (
            check_df["last_chemo_date"].notna() &
            ((check_df["effective_last_infusion"] - check_df["last_chemo_date"]).dt.days.abs()
             <= CHEMO_PROXIMITY_EXCLUSION_DAYS)
        )
        exclude_ids = set(check_df[check_df["chemo_near_end"]]["mpi_id"])
        eligible_ids -= exclude_ids
        attrition["after_chemo_exclusion"] = len(eligible_ids)
    else:
        attrition["after_chemo_exclusion"] = len(eligible_ids)

    # ── Step 7: Assign cohorts ──────────────────────────────────────────
    cohort_data = infusion_summary_eligible[
        infusion_summary_eligible["mpi_id"].isin(eligible_ids)
    ].copy()

    cohort_data["cohort"] = np.where(
        cohort_data["months_on_tx"] <= FIXED_DURATION_UPPER_MONTHS,
        "Fixed-Duration",
        "Continuation",
    )
    attrition["fixed_duration"] = (cohort_data["cohort"] == "Fixed-Duration").sum()
    attrition["continuation"] = (cohort_data["cohort"] == "Continuation").sum()

    # ── Step 8: Landmark at 29 months — must be alive ───────────────────
    cohort_data["landmark_date"] = cohort_data["start_date"] + pd.to_timedelta(
        LANDMARK_MONTHS * 30.44, unit="D"
    )

    # Merge death date and last activity from demographics
    demo_surv = demo[["mpi_id", "last_activity"]].copy()
    if "date_death" in demo.columns:
        demo_surv["date_death"] = demo["date_death"]
    else:
        demo_surv["date_death"] = pd.NaT
    cohort_data = cohort_data.merge(demo_surv, on="mpi_id", how="left")

    # Alive at landmark:
    #   - If patient died (date_death not null), they must have died AFTER landmark
    #   - If patient alive (date_death is null), last_activity must be >= landmark
    cohort_data["alive_at_landmark"] = np.where(
        cohort_data["date_death"].notna(),
        cohort_data["date_death"] >= cohort_data["landmark_date"],    # died after landmark
        cohort_data["last_activity"] >= cohort_data["landmark_date"], # alive, seen after landmark
    )
    cohort_data = cohort_data[cohort_data["alive_at_landmark"]].copy()
    eligible_ids = set(cohort_data["mpi_id"])
    attrition["alive_at_landmark"] = len(eligible_ids)

    # ── Step 9: Derive survival outcomes ────────────────────────────────
    # OS definition:
    #   - Event (os_event=1): patient has a date_death that is not null
    #   - Time: from landmark_date to date_death (if died) or last_activity (if censored)
    #   - Censoring: at min(last_activity, FOLLOWUP_CUTOFF) if no death recorded
    followup = pd.Timestamp(FOLLOWUP_CUTOFF)

    # Determine event indicator: died = date_death is not null
    cohort_data["os_event"] = cohort_data["date_death"].notna().astype(int)

    # Determine OS end date:
    #   - If died: date_death (capped at follow-up cutoff)
    #   - If alive/censored: min(last_activity, follow-up cutoff)
    cohort_data["os_end_date"] = np.where(
        cohort_data["date_death"].notna(),
        cohort_data["date_death"].clip(upper=followup),
        cohort_data["last_activity"].clip(upper=followup),
    )
    cohort_data["os_end_date"] = pd.to_datetime(cohort_data["os_end_date"])

    # OS time in days from landmark
    cohort_data["os_time_days"] = (
        cohort_data["os_end_date"] - cohort_data["landmark_date"]
    ).dt.days

    # Clamp negative values (shouldn't happen but safety)
    cohort_data["os_time_days"] = cohort_data["os_time_days"].clip(lower=0)

    cohort_data["os_time_months"] = cohort_data["os_time_days"] / 30.44

    # ── Step 10: Merge covariates ───────────────────────────────────────
    # Age at index (landmark date)
    disease_info = disease[["mpi_id", "histology", "metastatic_date", "diag_date"]].drop_duplicates("mpi_id")
    cohort_data = cohort_data.merge(disease_info, on="mpi_id", how="left")

    demo_cov = demo[["mpi_id", "age_dx", "gender", "race", "payer", "smoking_history"]].drop_duplicates("mpi_id")
    cohort_data = cohort_data.merge(demo_cov, on="mpi_id", how="left")

    # Age at index
    cohort_data["age_at_index"] = (
        cohort_data["age_dx"] +
        (cohort_data["landmark_date"] - cohort_data["diag_date"]).dt.days / 365.25
    ).round(1)

    # De novo vs recurrent
    cohort_data["de_novo_vs_recurrent"] = np.where(
        (cohort_data["metastatic_date"] - cohort_data["diag_date"]).dt.days <= 30,
        "De novo", "Recurrent"
    )

    # Pembro with chemo
    cohort_data["pembro_with_chemo"] = np.where(
        cohort_data["regimen"].str.contains(r"\+", na=False),
        "With Chemo", "Monotherapy"
    )

    # ECOG performance status
    # Primary source: LABS table (test_name = "ECOG", value = float 0–4)
    # Fallback: RISKSCORES table (risk_name contains "ECOG", value = string)
    ecog_ps = None

    if not labs.empty and "test_name" in labs.columns:
        ecog = labs[labs["test_name"].str.upper().str.contains("ECOG", na=False)].copy()
        if not ecog.empty:
            ecog = ecog.sort_values("test_date")
            ecog_latest = ecog.drop_duplicates("mpi_id", keep="last")[["mpi_id", "value"]]
            # value is float64 (0.0, 1.0, 2.0...) — convert to string category
            ecog_latest["ecog_ps"] = ecog_latest["value"].apply(
                lambda x: str(int(x)) if pd.notna(x) else "Unknown"
            )
            ecog_ps = ecog_latest[["mpi_id", "ecog_ps"]]

    if ecog_ps is None and not riskscores.empty:
        ecog = riskscores[riskscores["risk_name"].str.upper().str.contains("ECOG", na=False)].copy()
        if not ecog.empty:
            ecog = ecog.sort_values("test_date")
            ecog_latest = ecog.drop_duplicates("mpi_id", keep="last")[["mpi_id", "value"]]
            ecog_latest.rename(columns={"value": "ecog_ps"}, inplace=True)
            ecog_ps = ecog_latest

    if ecog_ps is not None:
        cohort_data = cohort_data.merge(ecog_ps, on="mpi_id", how="left")
        cohort_data["ecog_ps"] = cohort_data["ecog_ps"].fillna("Unknown")
    else:
        cohort_data["ecog_ps"] = "Unknown"

    # PD-L1 status
    if not biomarker.empty:
        pdl1 = biomarker[biomarker["biomarker_name"].str.upper().str.contains("PD-L1", na=False)].copy()
        if not pdl1.empty:
            pdl1_closest = pdl1.sort_values("test_date").drop_duplicates("mpi_id", keep="last")
            pdl1_closest = pdl1_closest[["mpi_id", "test_result"]].rename(
                columns={"test_result": "pdl1_status"}
            )
            cohort_data = cohort_data.merge(pdl1_closest, on="mpi_id", how="left")
        else:
            cohort_data["pdl1_status"] = "Unknown"
    else:
        cohort_data["pdl1_status"] = "Unknown"

    # Brain metastases
    if not metastases.empty:
        brain = metastases[
            metastases["metastatic_site"].str.lower().str.contains("brain", na=False)
        ]
        brain_ids = set(brain["mpi_id"].unique())
        cohort_data["brain_mets"] = np.where(
            cohort_data["mpi_id"].isin(brain_ids), "Yes", "No"
        )
    else:
        cohort_data["brain_mets"] = "Unknown"

    # Clean up
    final_cols = [
        "mpi_id", "cohort", "landmark_date", "start_date",
        "effective_last_infusion", "months_on_tx", "n_infusions",
        "os_time_days", "os_time_months", "os_event",
        "age_at_index", "gender", "race", "payer", "smoking_history",
        "ecog_ps", "pdl1_status", "histology", "de_novo_vs_recurrent",
        "brain_mets", "pembro_with_chemo", "regimen",
    ]
    # Only keep columns that exist
    final_cols = [c for c in final_cols if c in cohort_data.columns]
    cohort_df = cohort_data[final_cols].copy().reset_index(drop=True)

    return cohort_df, attrition


def print_attrition(attrition: dict) -> None:
    """Pretty-print the attrition table."""
    print("\n" + "=" * 60)
    print("ATTRITION TABLE")
    print("=" * 60)
    labels = {
        "total_patients":         "Total patients in dataset",
        "metastatic_nsclc_c34x":  "Metastatic NSCLC (C34.X)",
        "diagnosis_window":       "Diagnosed 2016-01 to 2025-08",
        "pembro_1l_metastatic":   "Pembrolizumab in 1L metastatic",
        "has_infusion_data":      "With infusion-level data",
        "infusion_at_22_26_months": "Infusion at ≥22 months",
        "after_chemo_exclusion":  "After chemo proximity exclusion",
        "fixed_duration":         "  → Fixed-Duration cohort",
        "continuation":           "  → Continuation cohort",
        "alive_at_landmark":      "Alive at 29-month landmark",
    }
    for key, label in labels.items():
        if key in attrition:
            print(f"  {label:40s} {attrition[key]:>6,}")
    print("=" * 60)

## just update
