"""
run_km_figure.py — Regenerate the KM OS figure only
====================================================
Run from the project folder:

    python run_km_figure.py               # 300 DPI (default)
    python run_km_figure.py --dpi 600     # 600 DPI for abstract submission

Loads cohort from parquet cache (fast — no full data reload needed).
Output saved to the same output folder as the full analysis.
"""

import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import load_tables
from cohort_selection import select_cohort
from analysis import run_kaplan_meier
from reporting import plot_km_curves


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dpi", type=int, default=300,
                   help="Output resolution (default 300; use 600 for abstract submission)")
    p.add_argument("--filename", type=str, default="km_os_landmark.png",
                   help="Output filename (default: km_os_landmark.png)")
    args = p.parse_args()

    print("=" * 60)
    print("KM FIGURE — regenerating from cache")
    print(f"DPI: {args.dpi}")
    print("=" * 60)

    # Load only the tables needed for cohort selection (from parquet cache)
    required = ["demographics", "disease", "lot", "dose",
                "biomarker", "vitals", "labs", "metastases",
                "comorbidities", "medicalcondition"]
    print("\nLoading tables from cache...")
    tables = load_tables(table_names=required, force_refresh=False)

    # Cohort selection
    print("Selecting cohort...")
    cohort_df, attrition = select_cohort(tables)
    print(f"  Cohort: {len(cohort_df)} patients  "
          f"(Fixed-Duration: {(cohort_df['cohort']=='Fixed-Duration').sum()}, "
          f"Continuation: {(cohort_df['cohort']=='Continuation').sum()})")

    # KM analysis
    print("Running Kaplan-Meier...")
    km_output = run_kaplan_meier(cohort_df)

    # Plot
    print("Generating figure...")
    outpath = plot_km_curves(km_output, output_filename=args.filename, dpi=args.dpi)

    print(f"\nDone. Figure saved to:\n  {outpath}")


if __name__ == "__main__":
    main()
