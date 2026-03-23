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

The script will:
  1. Load data (real CSVs or synthetic fallback)
  2. Apply SAP inclusion/exclusion criteria → analysis cohort
  3. Run Kaplan-Meier analysis + log-rank test
  4. Fit multivariate Cox PH model
  5. Generate all outputs (Table 1, KM plot, forest plot, attrition diagram)
"""

import sys
import warnings
warnings.filterwarnings("ignore")

# Ensure module imports work when running from any directory
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUTPUT_DIR, COVARIATES
from data_loader import load_tables
from cohort_selection import select_cohort, print_attrition
from analysis import (
    run_kaplan_meier, run_cox_model,
    print_km_summary, print_cox_summary,
)
from reporting import (
    generate_table1, plot_km_curves, plot_forest,
    plot_attrition, save_cohort_csv,
)


def main():
    print("=" * 70)
    print("PEMBROLIZUMAB FIXED-DURATION vs CONTINUATION — OS LANDMARK ANALYSIS")
    print("=" * 70)

    # ── 1. Load Data ────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    required_tables = [
        "demographics", "disease", "lot", "dose",
        "biomarker", "riskscores", "metastases",
    ]
    tables = load_tables(table_names=required_tables)

    # ── 2. Cohort Selection ─────────────────────────────────────────────
    print("\n[2/5] Applying inclusion/exclusion criteria...")
    cohort_df, attrition = select_cohort(tables)
    print_attrition(attrition)

    if len(cohort_df) == 0:
        print("\n⚠ No patients met all criteria. Check data and SAP parameters.")
        sys.exit(1)

    print(f"\nAnalysis cohort: {len(cohort_df)} patients")
    print(f"  Fixed-Duration: {(cohort_df['cohort'] == 'Fixed-Duration').sum()}")
    print(f"  Continuation:   {(cohort_df['cohort'] == 'Continuation').sum()}")

    # ── 3. Kaplan-Meier Analysis ────────────────────────────────────────
    print("\n[3/5] Running Kaplan-Meier survival analysis...")
    km_output = run_kaplan_meier(cohort_df)
    print_km_summary(km_output)

    # ── 4. Cox Proportional Hazards Model ───────────────────────────────
    print("\n[4/5] Fitting multivariate Cox PH model...")

    # First: univariate (treatment only)
    print("\n--- Univariate (treatment only) ---")
    cox_uni = run_cox_model(cohort_df, covariates=None)
    print_cox_summary(cox_uni)

    # Then: multivariate with available covariates
    available_covs = [c for c in COVARIATES if c in cohort_df.columns]
    if available_covs:
        print(f"\n--- Multivariate (adjusting for: {', '.join(available_covs)}) ---")
        cox_multi = run_cox_model(cohort_df, covariates=available_covs)
        print_cox_summary(cox_multi)
    else:
        print("\n[!] No covariates available for multivariate model")
        cox_multi = cox_uni

    # ── 5. Generate Reports ─────────────────────────────────────────────
    print("\n[5/5] Generating reports and figures...")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Table 1
    table1 = generate_table1(cohort_df)
    print(f"\n--- Table 1: Baseline Characteristics ---")
    print(table1.to_string(index=False))

    # KM curves
    plot_km_curves(km_output)

    # Forest plot
    plot_forest(cox_multi)

    # Attrition diagram
    plot_attrition(attrition)

    # Save analysis cohort
    save_cohort_csv(cohort_df)

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print(f"\nFiles generated:")
    for f in sorted(Path(OUTPUT_DIR).glob("*")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:45s} ({size_kb:.1f} KB)")

    print("\n✓ Pipeline finished successfully.")
    print("\nNOTE: This analysis used SYNTHETIC data for testing.")
    print("To run with real data:")
    print("  export PEMBRO_DATA_DIR=/path/to/IC_PrecisionQ_CSVs/")
    print("  python main.py")


if __name__ == "__main__":
    main()
