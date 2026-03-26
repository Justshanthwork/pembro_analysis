"""
main.py — Orchestrator for Pembrolizumab Fixed-Duration vs Continuation Analysis
=================================================================================
Run this script to execute the full analysis pipeline:

    python main.py

Or from VS Code terminal:
    cd pembro_analysis
    python main.py

To use real data, set the environment variable before running:
    export PEMBRO_DATA_DIR=/path/to/your/csv/folder
    python main.py

Full pipeline:
  1. Load data (real CSVs or synthetic fallback)
  2. Apply SAP inclusion/exclusion criteria → analysis cohort
  3. Run Kaplan-Meier OS analysis + log-rank test
  4. Generate Table 1 and KM plot
  5. Run Cox PH models (unadjusted → fully adjusted → LASSO)
  6. Schoenfeld residual PH assumption check
  7. Subgroup analyses + interaction tests
  8. Landmark sensitivity (27, 29, 32 months)
  9. Generate all reports and Excel workbook

Minimum CSVs required (copy these files):
  Core (cohort selection + OS):
    IC_PRECISIONQ_STN_DEATH_NSCLC_DEMOGRAPHICS_20260310_190335.csv
    IC_PRECISIONQ_STN_DEATH_NSCLC_DISEASE_20260310_190335.csv
    IC_PRECISIONQ_STN_DEATH_NSCLC_LOT_20260310_190335.csv
    IC_PRECISIONQ_STN_DEATH_NSCLC_DOSE_20260310_190335.csv
  Covariates (ECOG, PD-L1, brain mets):
    IC_PRECISIONQ_STN_DEATH_NSCLC_VITALS_20260310_190335.csv
    IC_PRECISIONQ_STN_DEATH_NSCLC_LABS_20260310_190335.csv
    IC_PRECISIONQ_STN_DEATH_NSCLC_RISKSCORES_20260310_190335.csv
    IC_PRECISIONQ_STN_DEATH_NSCLC_BIOMARKER_20260310_190335.csv
    IC_PRECISIONQ_STN_DEATH_NSCLC_METASTASES_20260310_190335.csv
  Extended covariates (comorbidities + medications):
    IC_PRECISIONQ_STN_DEATH_NSCLC_COMORBIDITIES_20260310_190335.csv
    IC_PRECISIONQ_STN_DEATH_NSCLC_MEDICALCONDITION_20260310_190335.csv
"""

import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

# Ensure module imports work when running from any directory
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    OUTPUT_DIR, FILES,
    COX_MODEL_SPECS, COVARIATES,
    SUBGROUP_VARIABLES, LANDMARK_SENSITIVITY_MONTHS,
    USE_LASSO_SELECTION, LASSO_ALPHA_RANGE,
)
from data_loader import load_tables, cache_status
from cohort_selection import select_cohort, print_attrition
from analysis import run_kaplan_meier, print_km_summary, build_km_supporting_table
from cox_analysis import (
    run_multiple_cox_models, run_lasso_cox,
    test_proportional_hazards, run_subgroup_analyses,
    run_landmark_sensitivity, build_model_comparison_table,
    print_model_comparison, print_subgroup_results,
    print_ph_test, print_landmark_sensitivity,
)
from reporting import (
    generate_table1, plot_km_curves,
    plot_attrition, save_cohort_csv,
    save_supporting_tables, save_methodology,
    plot_model_comparison_forest, plot_full_cox_forest,
    plot_subgroup_forest, plot_schoenfeld_residuals,
    plot_landmark_sensitivity, save_cox_tables,
)
from excel_report import create_excel_report


def _parse_args():
    p = argparse.ArgumentParser(
        description="Pembro Fixed-Duration vs Continuation OS Analysis"
    )
    p.add_argument(
        "--refresh", nargs="*", metavar="TABLE",
        help=(
            "Re-pull from Snowflake. No args = refresh all tables. "
            "Pass table names to refresh only those, e.g. --refresh dose demographics"
        ),
    )
    p.add_argument(
        "--cache-status", action="store_true",
        help="Show cache age/row counts and exit.",
    )
    return p.parse_args()


def main():
    args = _parse_args()

    if args.cache_status:
        cache_status()
        sys.exit(0)

    # Resolve force_refresh value
    if args.refresh is None:
        force_refresh = False          # use cache
    elif len(args.refresh) == 0:
        force_refresh = True           # --refresh with no args → all tables
    else:
        force_refresh = args.refresh   # --refresh dose demographics → specific tables

    print("=" * 70)
    print("PEMBROLIZUMAB FIXED-DURATION vs CONTINUATION — OS LANDMARK ANALYSIS")
    print("Phase: FULL (OS + Table 1 + Cox Models + Sensitivity)")
    print("=" * 70)

    # ── Required tables ───────────────────────────────────────────────────
    required_tables = [
        "demographics", "disease", "lot", "dose",
        "biomarker", "vitals", "labs", "metastases",
        "comorbidities", "medicalcondition",
    ]

    # ── 1. Load Data ──────────────────────────────────────────────────────
    print("\n[1/8] Loading data...")
    tables = load_tables(table_names=required_tables, force_refresh=force_refresh)

    # ── 2. Cohort Selection ───────────────────────────────────────────────
    print("\n[2/8] Applying SAP inclusion/exclusion criteria...")
    cohort_df, attrition = select_cohort(tables)
    print_attrition(attrition)

    if len(cohort_df) == 0:
        print("\n⚠ No patients met all criteria. Check data and SAP parameters.")
        sys.exit(1)

    print(f"\nAnalysis cohort: {len(cohort_df)} patients")
    print(f"  Fixed-Duration: {(cohort_df['cohort'] == 'Fixed-Duration').sum()}")
    print(f"  Continuation:   {(cohort_df['cohort'] == 'Continuation').sum()}")

    # ── 3. Kaplan-Meier OS Analysis ───────────────────────────────────────
    print("\n[3/8] Running Kaplan-Meier survival analysis...")
    km_output = run_kaplan_meier(cohort_df)
    print_km_summary(km_output)
    km_supporting = build_km_supporting_table(cohort_df, km_output)

    # ── 4. Table 1 & KM Reports ──────────────────────────────────────────
    print("\n[4/8] Generating Table 1 and KM reports...")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    table1 = generate_table1(cohort_df)
    print(f"\n--- Table 1: Baseline Characteristics ---")
    print(table1.to_string(index=False))

    plot_km_curves(km_output)
    save_supporting_tables(km_supporting, km_output)
    save_methodology(cohort_df, attrition, km_output)
    plot_attrition(attrition)
    save_cohort_csv(cohort_df)

    # ── 5. Cox Proportional Hazards Models ────────────────────────────────
    print("\n[5/8] Running Cox PH models...")
    print("=" * 70)

    cox_results = run_multiple_cox_models(cohort_df, COX_MODEL_SPECS)

    # LASSO-selected model
    lasso_result = None
    if USE_LASSO_SELECTION:
        print("\n  Fitting LASSO-selected Cox model...")
        lasso_result = run_lasso_cox(
            cohort_df, COVARIATES,
            alpha_range=LASSO_ALPHA_RANGE,
        )

    # Build comparison table
    comparison_df = build_model_comparison_table(cox_results, lasso_result)
    print_model_comparison(comparison_df)

    # ── 6. Schoenfeld Residuals — PH Assumption ──────────────────────────
    print("\n[6/8] Testing proportional hazards assumption...")

    # Use the "full" model covariates for PH testing
    ph_covariates = COX_MODEL_SPECS.get("full", COVARIATES)
    ph_result = test_proportional_hazards(cohort_df, ph_covariates)
    print_ph_test(ph_result)

    # ── 7. Subgroup Analyses ──────────────────────────────────────────────
    print("\n[7/8] Running subgroup analyses...")
    subgroup_df = run_subgroup_analyses(cohort_df, SUBGROUP_VARIABLES)
    print_subgroup_results(subgroup_df)

    # ── 8. Landmark Sensitivity ───────────────────────────────────────────
    print("\n[8/8] Running landmark sensitivity analysis...")

    # Use clinical model covariates for sensitivity (faster, more stable)
    sensitivity_covariates = COX_MODEL_SPECS.get("clinical", COVARIATES[:11])
    landmark_df = run_landmark_sensitivity(
        tables, select_cohort, LANDMARK_SENSITIVITY_MONTHS,
        sensitivity_covariates,
    )
    print_landmark_sensitivity(landmark_df)

    # ── Generate Reports ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GENERATING REPORTS AND FIGURES...")
    print("=" * 70)

    # Forest plots
    plot_model_comparison_forest(cox_results, lasso_result)

    # Full Cox forest plot (use "full" model if available, else "clinical")
    for model_key in ["full", "clinical"]:
        if model_key in cox_results and "error" not in cox_results[model_key]:
            plot_full_cox_forest(
                cox_results[model_key],
                model_name=model_key.title() + " Model",
                output_filename=f"forest_plot_{model_key}_cox.png",
            )
            break

    plot_subgroup_forest(subgroup_df)
    plot_schoenfeld_residuals(ph_result)
    plot_landmark_sensitivity(landmark_df)

    # Save CSV tables
    save_cox_tables(comparison_df, subgroup_df, ph_result, landmark_df,
                    cox_results, lasso_result)

    # ── Excel workbook ────────────────────────────────────────────────────
    print("\nBuilding Excel presentation workbook...")
    create_excel_report(
        cohort_df, attrition, km_output, km_supporting, table1,
        cox_results=cox_results,
        lasso_result=lasso_result,
        comparison_df=comparison_df,
        subgroup_df=subgroup_df,
        ph_result=ph_result,
        landmark_df=landmark_df,
    )

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print(f"\nFiles generated:")
    for f in sorted(Path(OUTPUT_DIR).glob("*")):
        if f.name.startswith("."):
            continue
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:50s} ({size_kb:.1f} KB)")

    print("\n✓ Full pipeline finished successfully.")


if __name__ == "__main__":
    main()
