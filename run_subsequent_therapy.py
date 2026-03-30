"""
run_subsequent_therapy.py — Subsequent therapy analysis (DOSE-based)
=====================================================================
Run from the project folder:

    python run_subsequent_therapy.py

Uses the DOSE table (individual drug administrations) to identify any
systemic therapy received after the 29-month landmark — more complete
than the LOT table which often only captures first-line data.

Key outputs:
  1. Rate of any post-landmark systemic therapy per cohort
  2. Drug-level breakdown (pembro rechallenge, chemo, IO, targeted)
  3. KM: time from landmark to first post-landmark drug (proxy PFS)
  4. OS sensitivity analysis censored at first post-landmark drug
  5. CSV + figure saved to output folder
"""

import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from config import OUTPUT_DIR, LANDMARK_MONTHS
from data_loader import load_tables
from cohort_selection import select_cohort

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

# ── Drug classification keywords ──────────────────────────────────────────────
PEMBRO_KEYWORDS   = ["pembrolizumab", "keytruda"]
IO_KEYWORDS       = ["nivolumab", "atezolizumab", "durvalumab", "ipilimumab",
                     "cemiplimab", "tremelimumab", "opdivo", "tecentriq",
                     "imfinzi", "libtayo"]
TARGETED_KEYWORDS = ["osimertinib", "erlotinib", "gefitinib", "afatinib",
                     "alectinib", "brigatinib", "lorlatinib", "crizotinib",
                     "capmatinib", "tepotinib", "selpercatinib", "pralsetinib",
                     "sotorasib", "adagrasib", "dabrafenib", "trametinib",
                     "tagrisso", "tarceva", "iressa", "xalkori", "alecensa"]
CHEMO_KEYWORDS    = ["carboplatin", "cisplatin", "pemetrexed", "paclitaxel",
                     "docetaxel", "gemcitabine", "etoposide", "vinorelbine",
                     "nab-paclitaxel", "abraxane"]


def _classify_drug(name: str) -> str:
    if not isinstance(name, str):
        return "Other / Unknown"
    n = name.lower()
    if any(k in n for k in PEMBRO_KEYWORDS):
        return "Pembro rechallenge"
    if any(k in n for k in IO_KEYWORDS):
        return "Other IO"
    if any(k in n for k in TARGETED_KEYWORDS):
        return "Targeted therapy"
    if any(k in n for k in CHEMO_KEYWORDS):
        return "Chemotherapy"
    return "Other / Unknown"


def main():
    print("=" * 65)
    print("SUBSEQUENT THERAPY ANALYSIS  (DOSE-based)")
    print("=" * 65)

    # ── Load data ─────────────────────────────────────────────────────────
    required = ["demographics", "disease", "lot", "dose",
                "biomarker", "vitals", "labs", "metastases",
                "comorbidities", "medicalcondition"]
    print("\nLoading tables from cache...")
    tables = load_tables(table_names=required, force_refresh=False)

    # ── Cohort ────────────────────────────────────────────────────────────
    print("Selecting cohort...")
    cohort_df, _ = select_cohort(tables)
    n_fd  = (cohort_df["cohort"] == "Fixed-Duration").sum()
    n_cnt = (cohort_df["cohort"] == "Continuation").sum()
    print(f"  Cohort: {len(cohort_df)} patients  "
          f"(Fixed-Duration: {n_fd}, Continuation: {n_cnt})")

    cohort_ref = cohort_df[["mpi_id", "cohort", "start_date",
                             "os_time_months", "os_event",
                             "effective_last_infusion"]].copy()
    cohort_ref["landmark_date"] = (
        cohort_ref["start_date"] + pd.to_timedelta(LANDMARK_MONTHS * 30.44, unit="D")
    )

    # ── Build post-landmark drug list from DOSE table ─────────────────────
    dose = tables["dose"].copy()
    dose.columns = dose.columns.str.lower().str.strip()
    dose["drug_exposure_start_date"] = pd.to_datetime(
        dose["drug_exposure_start_date"], errors="coerce"
    )

    # Keep only cohort patients
    dose = dose[dose["mpi_id"].isin(cohort_ref["mpi_id"])].copy()

    # Merge landmark date and effective_last_infusion per patient
    dose = dose.merge(
        cohort_ref[["mpi_id", "landmark_date", "cohort", "effective_last_infusion"]],
        on="mpi_id", how="inner"
    )

    # Post-landmark doses only
    dose_post = dose[dose["drug_exposure_start_date"] > dose["landmark_date"]].copy()

    # For continuation patients: exclude pembrolizumab doses that are just
    # their ongoing LOT 1 (i.e., pembro doses up to 60 days after last infusion)
    # We define "new" pembro as pembro starting >60 days after effective_last_infusion
    is_ongoing_pembro = (
        (dose_post["cohort"] == "Continuation") &
        dose_post["generic_name"].str.lower().str.contains("pembrolizumab", na=False) &
        (
            (dose_post["drug_exposure_start_date"] - dose_post["effective_last_infusion"])
            .dt.days <= 60
        )
    )
    dose_post = dose_post[~is_ongoing_pembro].copy()

    # Classify each drug
    dose_post["drug_name"] = dose_post["generic_name"].fillna(dose_post["brand_name"])
    dose_post["therapy_class"] = dose_post["drug_name"].apply(_classify_drug)

    print(f"\n  Post-landmark drug administrations found: {len(dose_post):,}")
    print(f"  Unique patients with any post-landmark drug: "
          f"{dose_post['mpi_id'].nunique():,}")

    # ── First post-landmark drug per patient ──────────────────────────────
    first_post = (
        dose_post.sort_values("drug_exposure_start_date")
        .groupby("mpi_id")
        .first()
        .reset_index()[["mpi_id", "drug_exposure_start_date", "therapy_class", "drug_name"]]
        .rename(columns={"drug_exposure_start_date": "subseq_date"})
    )

    cohort_ref = cohort_ref.merge(first_post, on="mpi_id", how="left")
    cohort_ref["had_subsequent"] = cohort_ref["subseq_date"].notna()
    cohort_ref["months_to_subseq"] = np.where(
        cohort_ref["had_subsequent"],
        (cohort_ref["subseq_date"] - cohort_ref["landmark_date"]).dt.days / 30.44,
        cohort_ref["os_time_months"],
    )
    cohort_ref["subseq_event"] = cohort_ref["had_subsequent"].astype(int)

    # ── Table 1: Rate of subsequent therapy ──────────────────────────────
    print("\n" + "=" * 65)
    print("TABLE 1: Post-landmark systemic therapy rate")
    print("=" * 65)
    rate_rows = []
    for cohort in ["Fixed-Duration", "Continuation"]:
        sub = cohort_ref[cohort_ref["cohort"] == cohort]
        n = len(sub)
        n_s = sub["had_subsequent"].sum()
        rate_rows.append({
            "Cohort": cohort,
            "N": n,
            "Any post-landmark therapy, n (%)": f"{n_s} ({n_s/n*100:.1f}%)",
        })
    rate_df = pd.DataFrame(rate_rows)
    print(rate_df.to_string(index=False))

    # ── Table 2: Drug class breakdown ────────────────────────────────────
    print("\n" + "=" * 65)
    print("TABLE 2: Post-landmark therapy class (first subsequent drug)")
    print("=" * 65)
    classes = ["Pembro rechallenge", "Other IO", "Targeted therapy",
               "Chemotherapy", "Other / Unknown"]
    type_rows = []
    for tc in classes:
        row = {"Therapy Class": tc}
        for cohort in ["Fixed-Duration", "Continuation"]:
            sub = cohort_ref[cohort_ref["cohort"] == cohort]
            n_total = len(sub)
            n_type = (sub["therapy_class"] == tc).sum()
            row[cohort] = f"{n_type} ({n_type/n_total*100:.1f}%)"
        type_rows.append(row)
    type_df = pd.DataFrame(type_rows)
    print(type_df.to_string(index=False))

    # ── Table 3: Top individual drugs post-landmark ───────────────────────
    print("\n" + "=" * 65)
    print("TABLE 3: Top 15 individual drugs post-landmark (all patients)")
    print("=" * 65)
    top_drugs = (
        dose_post.drop_duplicates(["mpi_id", "drug_name"])
        ["drug_name"].value_counts().head(15)
    )
    print(top_drugs.to_string())

    # ── Figures ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = {"Fixed-Duration": "#2166AC", "Continuation": "#B2182B"}
    legend_labels = {
        "Fixed-Duration": "Fixed-duration pembro\n(22–26 months)",
        "Continuation":   "Continued pembro\n(>26 months)",
    }

    # Left: Time to first post-landmark drug
    ax = axes[0]
    for cohort in sorted(cohort_ref["cohort"].unique()):
        sub = cohort_ref[cohort_ref["cohort"] == cohort]
        kmf = KaplanMeierFitter()
        kmf.fit(sub["months_to_subseq"], sub["subseq_event"],
                label=legend_labels.get(cohort, cohort))
        kmf.plot_survival_function(ax=ax, color=colors[cohort],
                                   linewidth=2, ci_alpha=0.15)

    fd  = cohort_ref[cohort_ref["cohort"] == "Fixed-Duration"]
    cnt = cohort_ref[cohort_ref["cohort"] == "Continuation"]
    lr_sub = logrank_test(fd["months_to_subseq"], cnt["months_to_subseq"],
                          fd["subseq_event"], cnt["subseq_event"])
    p_txt = (f"Log-rank p = {lr_sub.p_value:.4f}"
             if lr_sub.p_value >= 0.0001 else "Log-rank p < 0.0001")
    ax.text(0.03, 0.08, p_txt, transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.8))
    ax.set_title("Time to First Post-Landmark Therapy\nfrom 29-Month Landmark",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Months from Landmark", fontsize=9, fontweight="bold")
    ax.set_ylabel("Proportion without subsequent therapy", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(fontsize=8, frameon=True, framealpha=0.9)

    # Right: OS sensitivity — censor at first post-landmark drug
    ax2 = axes[1]
    cohort_ref["os_sens_time"] = np.where(
        cohort_ref["had_subsequent"],
        cohort_ref["months_to_subseq"],
        cohort_ref["os_time_months"],
    )
    cohort_ref["os_sens_event"] = np.where(
        cohort_ref["had_subsequent"], 0, cohort_ref["os_event"]
    )

    for cohort in sorted(cohort_ref["cohort"].unique()):
        sub = cohort_ref[cohort_ref["cohort"] == cohort]
        kmf = KaplanMeierFitter()
        kmf.fit(sub["os_sens_time"], sub["os_sens_event"],
                label=legend_labels.get(cohort, cohort))
        kmf.plot_survival_function(ax=ax2, color=colors[cohort],
                                   linewidth=2, ci_alpha=0.15)

    lr_os = logrank_test(fd["os_sens_time"], cnt["os_sens_time"],
                         fd["os_sens_event"], cnt["os_sens_event"])
    p_txt2 = (f"Log-rank p = {lr_os.p_value:.4f}"
              if lr_os.p_value >= 0.0001 else "Log-rank p < 0.0001")
    ax2.text(0.03, 0.08, p_txt2, transform=ax2.transAxes, fontsize=9, va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="gray", alpha=0.8))
    ax2.set_title("OS Sensitivity Analysis\n(Censored at Subsequent Therapy)",
                  fontsize=10, fontweight="bold")
    ax2.set_xlabel("Months from Landmark", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Overall Survival Probability", fontsize=9, fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(labelsize=8)
    ax2.grid(True, alpha=0.25, linestyle="--")
    ax2.legend(fontsize=8, frameon=True, framealpha=0.9)

    plt.tight_layout()
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    fig_path = Path(OUTPUT_DIR) / "subsequent_therapy_analysis.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nFigure saved to: {fig_path}")

    # ── Save CSVs ─────────────────────────────────────────────────────────
    rate_df.to_csv(Path(OUTPUT_DIR) / "subseq_therapy_rate.csv", index=False)
    type_df.to_csv(Path(OUTPUT_DIR) / "subseq_therapy_type.csv", index=False)
    cohort_ref[["mpi_id", "cohort", "had_subsequent", "therapy_class",
                "drug_name", "months_to_subseq", "subseq_event"]].to_csv(
        Path(OUTPUT_DIR) / "subseq_therapy_detail.csv", index=False
    )
    print("CSVs saved to output folder.")

    print("\n" + "=" * 65)
    print("INTERPRETATION GUIDE")
    print("=" * 65)
    print("""
Key questions:
  1. Fixed-Duration rate >> Continuation rate?
     → Subsequent therapy may partly explain equivalent OS.

  2. Pembro rechallenge common in Fixed-Duration?
     → Patients effectively continued IO exposure — narrows true difference.

  3. OS sensitivity (right panel) — curves diverge after censoring?
     → Subsequent therapy was masking a survival difference.
     → If still equivalent: no difference even before salvage therapy.
""")


if __name__ == "__main__":
    main()
