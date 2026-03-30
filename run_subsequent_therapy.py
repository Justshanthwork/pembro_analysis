"""
run_subsequent_therapy.py — Subsequent therapy analysis
========================================================
Run from the project folder:

    python run_subsequent_therapy.py

Addresses the question:
  Did fixed-duration patients receive more subsequent therapy after the
  29-month landmark, potentially confounding the equivalent OS finding?

Key outputs:
  1. Rate of any subsequent therapy per cohort
  2. Therapy type breakdown (pembro rechallenge, chemo, IO, targeted, combo)
  3. Kaplan-Meier of time from landmark to subsequent therapy (proxy for PFS)
  4. OS sensitivity analysis censored at subsequent systemic therapy
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

# ── Therapy classification keywords ──────────────────────────────────────────
PEMBRO_KEYWORDS    = ["pembrolizumab", "keytruda"]
IO_KEYWORDS        = ["nivolumab", "atezolizumab", "durvalumab", "ipilimumab",
                      "cemiplimab", "tremelimumab", "opdivo", "tecentriq",
                      "imfinzi", "libtayo"]
TARGETED_KEYWORDS  = ["osimertinib", "erlotinib", "gefitinib", "afatinib",
                      "alectinib", "brigatinib", "lorlatinib", "crizotinib",
                      "capmatinib", "tepotinib", "selpercatinib", "pralsetinib",
                      "sotorasib", "adagrasib", "dabrafenib", "trametinib",
                      "tagrisso", "tarceva", "iressa", "xalkori", "alecensa"]
CHEMO_KEYWORDS     = ["carboplatin", "cisplatin", "pemetrexed", "paclitaxel",
                      "docetaxel", "gemcitabine", "etoposide", "vinorelbine",
                      "nab-paclitaxel", "abraxane"]


def _classify_regimen(regimen: str) -> str:
    """Classify a regimen string into a therapy category."""
    if not isinstance(regimen, str):
        return "Other / Unknown"
    r = regimen.lower()
    has_pembro    = any(k in r for k in PEMBRO_KEYWORDS)
    has_other_io  = any(k in r for k in IO_KEYWORDS)
    has_targeted  = any(k in r for k in TARGETED_KEYWORDS)
    has_chemo     = any(k in r for k in CHEMO_KEYWORDS)

    if has_pembro and has_chemo:
        return "Pembro rechallenge + chemo"
    if has_pembro:
        return "Pembro rechallenge (mono)"
    if has_other_io and has_chemo:
        return "Other IO + chemo"
    if has_other_io:
        return "Other IO (mono)"
    if has_targeted:
        return "Targeted therapy"
    if has_chemo:
        return "Chemotherapy"
    return "Other / Unknown"


def main():
    print("=" * 65)
    print("SUBSEQUENT THERAPY ANALYSIS")
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
    print(f"  Cohort: {len(cohort_df)} patients  "
          f"(Fixed-Duration: {(cohort_df['cohort']=='Fixed-Duration').sum()}, "
          f"Continuation: {(cohort_df['cohort']=='Continuation').sum()})")

    # Each patient's landmark date = start_date + 29 months
    cohort_ref = cohort_df[["mpi_id", "cohort", "start_date",
                             "os_time_months", "os_event"]].copy()
    cohort_ref["landmark_date"] = (
        cohort_ref["start_date"] + pd.to_timedelta(LANDMARK_MONTHS * 30.44, unit="D")
    )

    # ── Identify subsequent therapies after the landmark ──────────────────
    lot = tables["lot"].copy()
    lot.columns = lot.columns.str.lower().str.strip()
    lot["start_date"] = pd.to_datetime(lot["start_date"], errors="coerce")

    # Keep only patients in our cohort
    lot = lot[lot["mpi_id"].isin(cohort_ref["mpi_id"])].copy()

    # Merge landmark date
    lot = lot.merge(cohort_ref[["mpi_id", "landmark_date", "cohort",
                                 "start_date", "os_time_months", "os_event"]],
                    on="mpi_id", how="inner",
                    suffixes=("", "_index"))

    # For continuation patients: their LOT 1 pembrolizumab extends past landmark —
    # exclude it (we want NEW therapies started after landmark, not the ongoing LOT 1)
    lot_subsequent = lot[
        (lot["start_date"] > lot["landmark_date"]) &
        ~(
            (lot["cohort"] == "Continuation") &
            (lot["lot"].astype(str).str.strip() == "1")
        )
    ].copy()

    lot_subsequent["therapy_class"] = lot_subsequent["regimen"].apply(_classify_regimen)

    # ── Time from landmark to first subsequent therapy ────────────────────
    first_subsequent = (
        lot_subsequent.sort_values("start_date")
        .groupby("mpi_id")
        .first()
        .reset_index()[["mpi_id", "start_date", "therapy_class", "regimen"]]
        .rename(columns={"start_date": "subseq_date"})
    )

    cohort_ref = cohort_ref.merge(first_subsequent, on="mpi_id", how="left")
    cohort_ref["had_subsequent"] = cohort_ref["subseq_date"].notna()

    # Months from landmark to subsequent therapy
    cohort_ref["months_to_subseq"] = np.where(
        cohort_ref["had_subsequent"],
        (cohort_ref["subseq_date"] - cohort_ref["landmark_date"]).dt.days / 30.44,
        cohort_ref["os_time_months"],   # censored at OS time if no subsequent therapy
    )
    cohort_ref["subseq_event"] = cohort_ref["had_subsequent"].astype(int)

    # ── Summary table 1: Rate of subsequent therapy ───────────────────────
    print("\n" + "=" * 65)
    print("TABLE: Subsequent Therapy Rate by Cohort")
    print("=" * 65)
    rate_rows = []
    for cohort in ["Fixed-Duration", "Continuation"]:
        sub = cohort_ref[cohort_ref["cohort"] == cohort]
        n = len(sub)
        n_subseq = sub["had_subsequent"].sum()
        rate_rows.append({
            "Cohort": cohort,
            "N": n,
            "Any subsequent therapy, n (%)": f"{n_subseq} ({n_subseq/n*100:.1f}%)",
        })
    rate_df = pd.DataFrame(rate_rows)
    print(rate_df.to_string(index=False))

    # ── Summary table 2: Therapy type breakdown ───────────────────────────
    print("\n" + "=" * 65)
    print("TABLE: Subsequent Therapy Type by Cohort")
    print("=" * 65)
    type_rows = []
    therapy_classes = [
        "Pembro rechallenge (mono)", "Pembro rechallenge + chemo",
        "Other IO (mono)", "Other IO + chemo",
        "Targeted therapy", "Chemotherapy", "Other / Unknown",
    ]
    for tc in therapy_classes:
        row = {"Therapy Type": tc}
        for cohort in ["Fixed-Duration", "Continuation"]:
            sub = cohort_ref[cohort_ref["cohort"] == cohort]
            n_total = len(sub)
            n_type = (sub["therapy_class"] == tc).sum()
            row[cohort] = f"{n_type} ({n_type/n_total*100:.1f}%)"
        type_rows.append(row)
    type_df = pd.DataFrame(type_rows)
    print(type_df.to_string(index=False))

    # ── KM: Time to subsequent therapy (proxy PFS) ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = {"Fixed-Duration": "#2166AC", "Continuation": "#B2182B"}
    legend_labels = {
        "Fixed-Duration": "Fixed-duration pembro\n(22–26 months)",
        "Continuation":   "Continued pembro\n(>26 months)",
    }

    # Left panel: Time to subsequent therapy
    ax = axes[0]
    km_subseq = {}
    for cohort in sorted(cohort_ref["cohort"].unique()):
        sub = cohort_ref[cohort_ref["cohort"] == cohort]
        kmf = KaplanMeierFitter()
        kmf.fit(sub["months_to_subseq"], sub["subseq_event"],
                label=legend_labels.get(cohort, cohort))
        kmf.plot_survival_function(ax=ax, color=colors[cohort],
                                   linewidth=2, ci_alpha=0.15)
        km_subseq[cohort] = kmf

    # Log-rank
    fd  = cohort_ref[cohort_ref["cohort"] == "Fixed-Duration"]
    cnt = cohort_ref[cohort_ref["cohort"] == "Continuation"]
    lr_sub = logrank_test(fd["months_to_subseq"], cnt["months_to_subseq"],
                          fd["subseq_event"], cnt["subseq_event"])
    p_txt = (f"Log-rank p = {lr_sub.p_value:.4f}"
             if lr_sub.p_value >= 0.0001 else "Log-rank p < 0.0001")
    ax.text(0.03, 0.08, p_txt, transform=ax.transAxes, fontsize=9,
            va="bottom", bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                   edgecolor="gray", alpha=0.8))
    ax.set_title("Time to Subsequent Therapy\nfrom 29-Month Landmark",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Months from Landmark", fontsize=9, fontweight="bold")
    ax.set_ylabel("Proportion without subsequent therapy", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(fontsize=8, frameon=True, framealpha=0.9)

    # Right panel: OS sensitivity — censor at subsequent therapy
    ax2 = axes[1]
    cohort_ref["os_sens_time"] = np.where(
        cohort_ref["had_subsequent"],
        cohort_ref["months_to_subseq"],   # censor at subsequent therapy
        cohort_ref["os_time_months"],
    )
    cohort_ref["os_sens_event"] = np.where(
        cohort_ref["had_subsequent"],
        0,   # censored
        cohort_ref["os_event"],
    )

    for cohort in sorted(cohort_ref["cohort"].unique()):
        sub = cohort_ref[cohort_ref["cohort"] == cohort]
        kmf = KaplanMeierFitter()
        kmf.fit(sub["os_sens_time"], sub["os_sens_event"],
                label=legend_labels.get(cohort, cohort))
        kmf.plot_survival_function(ax=ax2, color=colors[cohort],
                                   linewidth=2, ci_alpha=0.15)

    fd2  = cohort_ref[cohort_ref["cohort"] == "Fixed-Duration"]
    cnt2 = cohort_ref[cohort_ref["cohort"] == "Continuation"]
    lr_os = logrank_test(fd2["os_sens_time"], cnt2["os_sens_time"],
                         fd2["os_sens_event"], cnt2["os_sens_event"])
    p_txt2 = (f"Log-rank p = {lr_os.p_value:.4f}"
              if lr_os.p_value >= 0.0001 else "Log-rank p < 0.0001")
    ax2.text(0.03, 0.08, p_txt2, transform=ax2.transAxes, fontsize=9,
             va="bottom", bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
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
    (cohort_ref[["mpi_id", "cohort", "had_subsequent", "therapy_class",
                 "months_to_subseq", "subseq_event"]]
     .to_csv(Path(OUTPUT_DIR) / "subseq_therapy_detail.csv", index=False))
    print("CSVs saved to output folder.")

    print("\n" + "=" * 65)
    print("INTERPRETATION GUIDE")
    print("=" * 65)
    print("""
Key questions this analysis answers:
  1. Did fixed-duration patients receive subsequent therapy at HIGHER rates?
     → If yes, subsequent therapy may partly explain equivalent OS.

  2. Was pembrolizumab rechallenge common in the fixed-duration group?
     → Rechallenge = ongoing IO exposure, narrowing the true difference.

  3. Does OS remain equivalent when censoring at subsequent therapy?
     → If OS diverges after censoring, subsequent therapy is confounding.
     → If OS remains equivalent, the groups are truly similar in survival.
""")


if __name__ == "__main__":
    main()
