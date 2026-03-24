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

Current phase — OS + Table 1 only (Cox model deferred):
  1. Load data (real CSVs or synthetic fallback)
  2. Apply SAP inclusion/exclusion criteria → analysis cohort
  3. Run Kaplan-Meier OS analysis + log-rank test
  4. Generate Table 1 and KM plot

Minimum CSVs required (copy only these 8 files):
  Core (cohort selection + OS):
    IC_PRECISIONQ_STN_DEATH_NSCLC_DEMOGRAPHICS_20260310_190335.csv
    IC_PRECISIONQ_STN_DEATH_NSCLC_DISEASE_20260310_190335.csv
    IC_PRECISIONQ_STN_DEATH_NSCLC_LOT_20260310_190335.csv
    IC_PRECISIONQ_STN_DEATH_NSCLC_DOSE_20260310_190335.csv
  Table 1 covariates (ECOG, PD-L1, brain mets):
    IC_PRECISIONQ_STN_DEATH_NSCLC_LABS_20260310_190335.csv       ← ECOG (test_name="ECOG", value=float)
    IC_PRECISIONQ_STN_DEATH_NSCLC_RISKSCORES_20260310_190335.csv ← ECOG fallback only
    IC_PRECISIONQ_STN_DEATH_NSCLC_BIOMARKER_20260310_190335.csv
    IC_PRECISIONQ_STN_DEATH_NSCLC_METASTASES_20260310_190335.csv
"""

import sys
import warnings
warnings.filterwarnings("ignore")

# Ensure module imports work when running from any directory
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUTPUT_DIR, FILES
from data_loader import load_tables
from cohort_selection import select_cohort, print_attrition
from analysis import run_kaplan_meier, print_km_summary, build_km_supporting_table
from reporting import (
    generate_table1, plot_km_curves,
    plot_attrition, save_cohort_csv,
    save_supporting_tables, save_methodology,
)
from excel_report import create_excel_report


def main():
    print("=" * 70)
    print("PEMBROLIZUMAB FIXED-DURATION vs CONTINUATION — OS LANDMARK ANALYSIS")
    print("Phase: OS + Table 1  (Cox deferred)")
    print("=" * 70)

    # ── Minimum required CSVs ────────────────────────────────────────────
    # Only these 7 tables are needed for this phase — no need to copy others.
    required_tables = [
        "demographics",   # age, sex, race, payer, smoking, death/last-activity
        "disease",        # cancer code, histology, diagnosis/metastatic dates
        "lot",            # line of therapy, regimen, start date
        "dose",           # individual infusion records (gap rule + chemo check)
        "biomarker",      # PD-L1 status (Table 1)
        "labs",           # ECOG PS (test_name = "ECOG", value = float; Table 1)
        "riskscores",     # ECOG fallback if not in LABS; can skip if LABS always has it
        "metastases",     # brain metastases (Table 1)
    ]

    # ── 1. Load Data ────────────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    print("Required CSVs for this phase:")
    for t in required_tables:
        print(f"  {FILES[t]}")
    print()
    tables = load_tables(table_names=required_tables)

    # ── 2. Cohort Selection ─────────────────────────────────────────────
    print("\n[2/4] Applying SAP inclusion/exclusion criteria...")
    cohort_df, attrition = select_cohort(tables)
    print_attrition(attrition)

    if len(cohort_df) == 0:
        print("\n⚠ No patients met all criteria. Check data and SAP parameters.")
        sys.exit(1)

    print(f"\nAnalysis cohort: {len(cohort_df)} patients")
    print(f"  Fixed-Duration: {(cohort_df['cohort'] == 'Fixed-Duration').sum()}")
    print(f"  Continuation:   {(cohort_df['cohort'] == 'Continuation').sum()}")

    # ── 3. Kaplan-Meier OS Analysis ─────────────────────────────────────
    print("\n[3/4] Running Kaplan-Meier survival analysis...")
    km_output = run_kaplan_meier(cohort_df)
    print_km_summary(km_output)

    # Build supporting KM table
    km_supporting = build_km_supporting_table(cohort_df, km_output)

    # ── 4. Generate Reports ─────────────────────────────────────────────
    print("\n[4/4] Generating reports and figures...")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Table 1
    table1 = generate_table1(cohort_df)
    print(f"\n--- Table 1: Baseline Characteristics ---")
    print(table1.to_string(index=False))

    # KM curves
    plot_km_curves(km_output)

    # Supporting KM tables
    save_supporting_tables(km_supporting, km_output)

    # Methodology summary
    save_methodology(cohort_df, attrition, km_output)

    # Attrition diagram
    plot_attrition(attrition)

    # Save analysis cohort
    save_cohort_csv(cohort_df)

    # Excel presentation workbook
    print("\n[4/4] Building Excel presentation workbook...")
    create_excel_report(cohort_df, attrition, km_output, km_supporting, table1)

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE COMPLETE: OS + Table 1")
    print("=" * 70)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print(f"\nFiles generated:")
    for f in sorted(Path(OUTPUT_DIR).glob("*")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:45s} ({size_kb:.1f} KB)")

    print("\n✓ Pipeline finished successfully.")
    print("\nNext step: run Cox model (requires the same 8 CSVs — no additional files needed).")


if __name__ == "__main__":
    main()
