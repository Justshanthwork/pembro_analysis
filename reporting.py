"""
reporting.py — Generate Table 1, KM curves, Forest Plot, Attrition Diagram
===========================================================================
All figures saved to OUTPUT_DIR as publication-ready PNGs and the analysis
tables as CSVs.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from config import OUTPUT_DIR


def ensure_output_dir():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 1 — Baseline characteristics
# ─────────────────────────────────────────────────────────────────────────────

def _summarize_categorical(df: pd.DataFrame, col: str, group_col: str) -> pd.DataFrame:
    """Frequency table for a categorical variable by group."""
    groups = sorted(df[group_col].unique())
    rows = []
    categories = df[col].fillna("Unknown").value_counts().index.tolist()

    for cat in categories:
        row = {"Variable": f"  {cat}"}
        for grp in groups:
            grp_data = df[df[group_col] == grp][col].fillna("Unknown")
            n = (grp_data == cat).sum()
            pct = n / len(grp_data) * 100 if len(grp_data) > 0 else 0
            row[grp] = f"{n} ({pct:.1f}%)"
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize_continuous(df: pd.DataFrame, col: str, group_col: str) -> pd.DataFrame:
    """Summary stats for a continuous variable by group."""
    groups = sorted(df[group_col].unique())
    row = {"Variable": f"  Median [IQR]"}
    for grp in groups:
        vals = df[df[group_col] == grp][col].dropna()
        if len(vals) > 0:
            med = vals.median()
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            row[grp] = f"{med:.1f} [{q1:.1f}–{q3:.1f}]"
        else:
            row[grp] = "—"
    return pd.DataFrame([row])


def generate_table1(
    cohort_df: pd.DataFrame,
    group_col: str = "cohort",
) -> pd.DataFrame:
    """
    Generate Table 1: Baseline Characteristics by Cohort.
    Returns a formatted DataFrame.
    """
    ensure_output_dir()
    groups = sorted(cohort_df[group_col].unique())
    all_rows = []

    # Header row with N
    header = {"Variable": "N"}
    for grp in groups:
        header[grp] = str((cohort_df[group_col] == grp).sum())
    all_rows.append(header)

    # Age
    all_rows.append({"Variable": "Age at landmark", **{g: "" for g in groups}})
    all_rows.extend(_summarize_continuous(cohort_df, "age_at_index", group_col).to_dict("records"))

    # Categorical variables
    cat_vars = {
        "Gender": "gender",
        "Race": "race",
        "Payer": "payer",
        "Smoking History": "smoking_history",
        "ECOG (<=1 vs 2+)": "ecog_binary",
        "PD-L1 Category": "pdl1_cat",
        "Histology Category": "histology_cat",
        "De Novo vs Recurrent": "de_novo_vs_recurrent",
        "Brain Metastases": "brain_mets",
        "Treatment Type": "pembro_with_chemo",
    }

    for label, col in cat_vars.items():
        if col not in cohort_df.columns:
            continue
        all_rows.append({"Variable": label, **{g: "" for g in groups}})
        all_rows.extend(_summarize_categorical(cohort_df, col, group_col).to_dict("records"))

    table1 = pd.DataFrame(all_rows)

    # Save
    table1.to_csv(OUTPUT_DIR / "table1_baseline_characteristics.csv", index=False, encoding="utf-8-sig")
    print(f"[reporting] Table 1 saved to {OUTPUT_DIR / 'table1_baseline_characteristics.csv'}")

    return table1


# ─────────────────────────────────────────────────────────────────────────────
# KAPLAN-MEIER SURVIVAL CURVES
# ─────────────────────────────────────────────────────────────────────────────

def plot_km_curves(
    km_output: dict,
    output_filename: str = "km_os_landmark.png",
    title: str = "Overall Survival from 29-Month Landmark\nFixed-Duration vs Continuation Pembrolizumab",
) -> Path:
    """
    Publication-quality KM plot with number-at-risk table.
    """
    ensure_output_dir()

    km_results = km_output["km_results"]
    lr = km_output["logrank"]

    # Color scheme
    colors = {
        "Fixed-Duration": "#2166AC",   # blue
        "Continuation": "#B2182B",     # red
    }

    # Descriptive legend labels shown on the figure
    legend_labels = {
        "Continuation":   "Continued pembrolizumab (>26 months)",
        "Fixed-Duration": "Fixed-duration pembrolizumab (22-26 months)",
    }

    # ── Layout: main plot (top) + at-risk table (bottom) ─────────────────
    # Use explicit figure + axes positioning so we can size each section
    # independently and avoid any overlap.
    fig = plt.figure(figsize=(11, 9))

    # Main KM axes: left=0.13, bottom=0.30, width=0.84, height=0.63
    ax = fig.add_axes([0.13, 0.30, 0.84, 0.63])

    # At-risk table axes: immediately below, same left/width, height=0.22
    n_groups = len(km_results)
    ax2 = fig.add_axes([0.13, 0.04, 0.84, min(0.22, 0.09 * n_groups + 0.06)])

    # ── Plot KM curves with descriptive labels ────────────────────────────
    for grp_name, kmf in sorted(km_results.items()):
        color = colors.get(grp_name, "gray")
        label = legend_labels.get(grp_name, grp_name)
        kmf.plot_survival_function(
            ax=ax,
            ci_show=True,
            color=color,
            linewidth=2,
            ci_alpha=0.15,
            label=label,
        )

    ax.set_xlabel("Time from Landmark (months)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Overall Survival Probability", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Legend: place in upper right to avoid overlapping the lower part of curves;
    # use bbox_to_anchor to keep it fully inside the axes.
    ax.legend(
        fontsize=10, loc="upper right",
        frameon=True, framealpha=0.9,
        bbox_to_anchor=(1.0, 1.0),
    )

    # ── Log-rank p-value: top-left box ───────────────────────────────────
    if lr is not None:
        p_text = f"Log-rank p = {lr.p_value:.4f}" if lr.p_value >= 0.0001 else "Log-rank p < 0.0001"
        ax.text(
            0.03, 0.08, p_text,
            transform=ax.transAxes, fontsize=11,
            verticalalignment="bottom", horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
        )

    # ── Median OS annotations: placed just above the at-risk table header,
    #    in the lower-left of the main axes so they don't crowd the curves.
    summary = km_output["summary"]
    y_pos = 0.22
    for _, row in summary.iterrows():
        cohort_short = legend_labels.get(row["Cohort"], row["Cohort"])
        text = f'{cohort_short}: Median {row["Median OS (months)"]:.1f} mo (95% CI {row["95% CI Lower"]:.1f}–{row["95% CI Upper"]:.1f})'
        ax.text(
            0.03, y_pos, text,
            transform=ax.transAxes, fontsize=8.5,
            verticalalignment="bottom", horizontalalignment="left",
            fontfamily="monospace",
            bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none", alpha=0.7),
        )
        y_pos -= 0.07

    # ── Number-at-risk table ──────────────────────────────────────────────
    time_points = np.arange(0, ax.get_xlim()[1], 6)
    risk_data = {}
    for grp_name, kmf in sorted(km_results.items()):
        n_at_risk = []
        for t in time_points:
            try:
                st = kmf.event_table
                nar = st.loc[st.index <= t, "at_risk"].iloc[-1] if len(st[st.index <= t]) > 0 else 0
            except Exception:
                nar = 0
            n_at_risk.append(int(nar))
        risk_data[grp_name] = n_at_risk

    ax2.set_xlim(ax.get_xlim())
    ax2.set_ylim(-0.5, n_groups + 0.5)
    ax2.axis("off")

    # "No. at risk" header row (topmost row of the at-risk block)
    ax2.text(
        -0.01, n_groups, "No. at risk",
        fontsize=9, fontweight="bold",
        ha="right", va="center",
        transform=ax2.get_yaxis_transform(),
    )

    for idx, (grp_name, counts) in enumerate(sorted(risk_data.items())):
        color = colors.get(grp_name, "gray")
        short_label = legend_labels.get(grp_name, grp_name)
        # Group label in the left margin (figure-level transform keeps it clear of data area)
        ax2.text(
            -0.01, idx, short_label,
            fontsize=8.5, fontweight="bold",
            ha="right", va="center", color=color,
            transform=ax2.get_yaxis_transform(),
        )
        for j, t in enumerate(time_points):
            ax2.text(
                t, idx, str(counts[j]),
                fontsize=8.5, ha="center", va="center", color=color,
            )

    outpath = OUTPUT_DIR / output_filename
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[reporting] KM plot saved to {outpath}")
    return outpath


# ─────────────────────────────────────────────────────────────────────────────
# FOREST PLOT — Cox model HRs
# ─────────────────────────────────────────────────────────────────────────────

def plot_forest(
    cox_output: dict,
    output_filename: str = "forest_plot_cox.png",
    title: str = "Multivariate Cox Proportional Hazards Model\nHazard Ratios with 95% CI",
) -> Path:
    """Forest plot showing HRs from Cox model."""
    ensure_output_dir()

    summary = cox_output["summary"].copy()
    summary = summary.sort_values("exp(coef)")

    labels = summary.index.tolist()
    hrs = summary["exp(coef)"].values
    lower = summary["exp(coef) lower 95%"].values
    upper = summary["exp(coef) upper 95%"].values
    pvals = summary["p"].values

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.5 + 2)))

    y_pos = np.arange(len(labels))

    # Plot HR points and CI lines
    for i in range(len(labels)):
        color = "#B2182B" if pvals[i] < 0.05 else "#666666"
        ax.plot([lower[i], upper[i]], [y_pos[i], y_pos[i]], color=color, linewidth=2)
        ax.plot(hrs[i], y_pos[i], "o", color=color, markersize=8, zorder=5)

    # Reference line at HR=1
    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Hazard Ratio (95% CI)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")

    # Add HR text on right
    for i in range(len(labels)):
        text = f"{hrs[i]:.2f} ({lower[i]:.2f}–{upper[i]:.2f}), p={pvals[i]:.3f}"
        ax.text(ax.get_xlim()[1] * 1.02, y_pos[i], text,
                fontsize=8, va="center", fontfamily="monospace")

    plt.tight_layout()
    outpath = OUTPUT_DIR / output_filename
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[reporting] Forest plot saved to {outpath}")
    return outpath


# ─────────────────────────────────────────────────────────────────────────────
# ATTRITION FLOW DIAGRAM
# ─────────────────────────────────────────────────────────────────────────────

def plot_attrition(
    attrition: dict,
    output_filename: str = "attrition_diagram.png",
) -> Path:
    """Simple attrition flow diagram."""
    ensure_output_dir()

    steps = [
        ("Total patients in dataset", attrition.get("total_patients", 0)),
        ("Metastatic NSCLC (C34.X)", attrition.get("metastatic_nsclc_c34x", 0)),
        ("Diagnosed 2016-01 to 2025-08", attrition.get("diagnosis_window", 0)),
        ("Pembrolizumab in 1L metastatic", attrition.get("pembro_1l_metastatic", 0)),
        ("Infusion at ≥22 months", attrition.get("infusion_at_22_26_months", 0)),
        ("After chemo exclusion", attrition.get("after_chemo_exclusion", 0)),
        ("Alive at 29-month landmark", attrition.get("alive_at_landmark", 0)),
    ]

    fig, ax = plt.subplots(figsize=(8, len(steps) * 1.2 + 2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(steps) * 1.5 + 1)
    ax.axis("off")
    ax.set_title("Patient Attrition Flow Diagram", fontsize=14, fontweight="bold", pad=20)

    box_width = 6
    box_height = 0.9
    x_center = 5

    for i, (label, count) in enumerate(steps):
        y = len(steps) * 1.5 - i * 1.5
        color = "#4393C3" if i < len(steps) - 1 else "#2166AC"

        # Box
        rect = plt.Rectangle(
            (x_center - box_width / 2, y - box_height / 2),
            box_width, box_height,
            linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.15,
        )
        ax.add_patch(rect)

        # Text
        ax.text(x_center, y, f"{label}\n(N = {count:,})",
                ha="center", va="center", fontsize=10, fontweight="bold")

        # Arrow
        if i < len(steps) - 1:
            excluded = steps[i][1] - steps[i + 1][1]
            ax.annotate(
                "", xy=(x_center, y - box_height / 2 - 0.1),
                xytext=(x_center, y - box_height / 2 - 0.5),
                arrowprops=dict(arrowstyle="<-", color="gray", lw=1.5),
            )
            if excluded > 0:
                ax.text(
                    x_center + box_width / 2 + 0.3,
                    y - box_height / 2 - 0.3,
                    f"Excluded: {excluded:,}",
                    fontsize=8, color="red", va="center",
                )

    # Final split into cohorts
    fd = attrition.get("fixed_duration", 0)
    cont = attrition.get("continuation", 0)
    y_bottom = len(steps) * 1.5 - (len(steps) - 1) * 1.5 - 1.5

    ax.text(x_center - 2, y_bottom, f"Fixed-Duration\n(N = {fd:,})",
            ha="center", va="center", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#2166AC", alpha=0.2))
    ax.text(x_center + 2, y_bottom, f"Continuation\n(N = {cont:,})",
            ha="center", va="center", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#B2182B", alpha=0.2))

    outpath = OUTPUT_DIR / output_filename
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[reporting] Attrition diagram saved to {outpath}")
    return outpath


# ─────────────────────────────────────────────────────────────────────────────
# SAVE ANALYSIS COHORT
# ─────────────────────────────────────────────────────────────────────────────

def save_supporting_tables(km_supporting: dict, km_output: dict) -> None:
    """Save KM supporting table and number-at-risk table as CSVs."""
    ensure_output_dir()

    # Summary table
    st = km_supporting["summary_table"]
    st.to_csv(OUTPUT_DIR / "km_supporting_table.csv", index=False, encoding="utf-8-sig")
    print(f"[reporting] KM supporting table saved to {OUTPUT_DIR / 'km_supporting_table.csv'}")
    print("\n--- KM Supporting Table ---")
    print(st.to_string(index=False))

    # Number at risk
    nar = km_supporting["number_at_risk"]
    nar.to_csv(OUTPUT_DIR / "km_number_at_risk.csv", index=False, encoding="utf-8-sig")
    print(f"\n[reporting] Number-at-risk table saved to {OUTPUT_DIR / 'km_number_at_risk.csv'}")
    print("\n--- Number at Risk ---")
    print(nar.to_string(index=False))


def save_methodology(cohort_df: pd.DataFrame, attrition: dict, km_output: dict) -> None:
    """Save a plain-text methodology summary to the output folder."""
    ensure_output_dir()
    from config import (
        DIAGNOSIS_START, DIAGNOSIS_END, FOLLOWUP_CUTOFF,
        FIXED_DURATION_LOWER_MONTHS, FIXED_DURATION_UPPER_MONTHS,
        CONTINUATION_LOWER_MONTHS, LANDMARK_MONTHS,
        MAX_INFUSION_GAP_DAYS, CHEMO_PROXIMITY_EXCLUSION_DAYS,
    )
    import datetime

    n_total   = attrition.get("total_patients", "N/A")
    n_final   = len(cohort_df)
    n_fd      = (cohort_df["cohort"] == "Fixed-Duration").sum()
    n_cont    = (cohort_df["cohort"] == "Continuation").sum()
    lr        = km_output["logrank"]
    p_val     = f"{lr.p_value:.4f}" if lr is not None else "N/A"

    lines = [
        "METHODOLOGY SUMMARY",
        "=" * 70,
        "",
        "Study Design",
        "  Retrospective observational cohort study using IC PrecisionQ real-world",
        "  electronic health record (EHR) data with a landmark analysis design.",
        "",
        "Data Source",
        "  IC PrecisionQ NSCLC dataset. Follow-up through 02/28/2026.",
        "",
        "Study Population",
        f"  Total patients in dataset:            {n_total:,}",
        f"  Final analysis cohort (post-landmark): {n_final:,}",
        f"    Fixed-Duration cohort:               {n_fd:,}",
        f"    Continuation cohort:                 {n_cont:,}",
        "",
        "Inclusion Criteria",
        "  - Age ≥18 years",
        "  - Metastatic NSCLC (ICD-10: C34.X), de novo or recurrent",
        f"  - Diagnosis date: {DIAGNOSIS_START} to {DIAGNOSIS_END}",
        "  - Pembrolizumab in first metastatic line of therapy (with or without chemo)",
        f"  - At least one pembrolizumab infusion at ≥{FIXED_DURATION_LOWER_MONTHS} months from treatment start",
        f"  - Alive at {LANDMARK_MONTHS}-month landmark",
        "",
        "Exclusion Criteria",
        f"  - Chemotherapy within {CHEMO_PROXIMITY_EXCLUSION_DAYS} days of last pembrolizumab infusion",
        "    (interpreted as progression-motivated treatment stop)",
        "",
        "Treatment Gap Rule",
        f"  If two consecutive pembrolizumab infusions were >{MAX_INFUSION_GAP_DAYS} days apart,",
        "  treatment was considered to have ended at the earlier infusion date.",
        "",
        "Exposure Groups",
        f"  Fixed-Duration: last infusion {FIXED_DURATION_LOWER_MONTHS}–{FIXED_DURATION_UPPER_MONTHS} months from treatment start",
        f"  Continuation:   last infusion >{CONTINUATION_LOWER_MONTHS} months from treatment start",
        "",
        "Landmark Definition",
        f"  {LANDMARK_MONTHS} months following pembrolizumab initiation.",
        "  Only patients alive and under observation at the landmark are included.",
        "",
        "Primary Endpoint",
        "  Overall survival (OS): time from landmark date to death from any cause.",
        f"  Patients alive at end of follow-up censored at date of last clinical contact",
        f"  (administrative censoring at {FOLLOWUP_CUTOFF}).",
        "",
        "Statistical Methods",
        "  - Kaplan-Meier method for survival estimation",
        "  - Log-rank test for between-group comparison (two-sided, α=0.05)",
        f"  - Log-rank p-value: {p_val}",
        "  - Median follow-up estimated using reverse Kaplan-Meier (Schemper & Smith method)",
        "  - Primary Cox proportional hazards model adjusted for age (continuous), race, ECOG (<=1 vs 2+),",
        "    PD-L1 category (Positive, Negative, Unknown/Not reported), histology category",
        "    (squamous, non-squamous including adenocarcinoma, unknown), and treatment type",
        "    (with or without chemotherapy)",
        "  - Stratified treatment HRs within levels of age group, race, ECOG, PD-L1, histology,",
        "    and treatment type (age grouped only for this stratified display)",
        "",
        "Software",
        "  Python (lifelines, pandas, matplotlib)",
        "",
        f"Report generated: {datetime.date.today().isoformat()}",
        "=" * 70,
    ]

    text = "\n".join(lines)
    print("\n" + text)

    out = OUTPUT_DIR / "methodology_summary.txt"
    out.write_text(text, encoding="utf-8")
    print(f"\n[reporting] Methodology summary saved to {out}")


def save_cohort_csv(cohort_df: pd.DataFrame, filename: str = "analysis_cohort.csv") -> Path:
    """Save the analysis-ready cohort to CSV."""
    ensure_output_dir()
    outpath = OUTPUT_DIR / filename
    cohort_df.to_csv(outpath, index=False)
    print(f"[reporting] Analysis cohort saved to {outpath}")
    return outpath


# ─────────────────────────────────────────────────────────────────────────────
# COX MODEL COMPARISON FOREST PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison_forest(
    cox_results: dict,
    lasso_result: dict = None,
    output_filename: str = "forest_model_comparison.png",
    title: str = "Treatment HR Across Cox Models\n(Continuation vs Fixed-Duration)",
) -> Path:
    """Forest plot comparing treatment HR across different model specifications."""
    ensure_output_dir()

    entries = []
    for model_name, result in cox_results.items():
        if "error" in result:
            continue
        hr = result["treatment_hr"]
        entries.append({
            "label": model_name.replace("_", " ").title(),
            "hr": hr["HR"],
            "lower": hr["HR_lower"],
            "upper": hr["HR_upper"],
            "p": hr["p_value"],
            "n": result["n_patients"],
        })

    if lasso_result and "error" not in lasso_result:
        hr = lasso_result["treatment_hr"]
        entries.append({
            "label": "LASSO-Selected",
            "hr": hr["HR"],
            "lower": hr["HR_lower"],
            "upper": hr["HR_upper"],
            "p": hr["p_value"],
            "n": lasso_result["n_patients"],
        })

    if not entries:
        print("[reporting] No valid models to plot.")
        return None

    fig, ax = plt.subplots(figsize=(10, max(3, len(entries) * 0.8 + 2)))

    y_pos = np.arange(len(entries))
    colors = ["#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "#B2182B"]

    for i, entry in enumerate(entries):
        color = colors[i % len(colors)]
        ax.plot([entry["lower"], entry["upper"]], [y_pos[i]] * 2,
                color=color, linewidth=2.5)
        ax.plot(entry["hr"], y_pos[i], "D", color=color, markersize=10, zorder=5)

    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([e["label"] for e in entries], fontsize=11)
    ax.set_xlabel("Hazard Ratio (95% CI)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")

    # Annotate with HR values
    for i, entry in enumerate(entries):
        text = f"HR {entry['hr']:.3f} ({entry['lower']:.3f}–{entry['upper']:.3f}), p={entry['p']:.4f}, N={entry['n']}"
        ax.text(ax.get_xlim()[1] * 1.05, y_pos[i], text,
                fontsize=8, va="center", fontfamily="monospace")

    plt.tight_layout()
    outpath = OUTPUT_DIR / output_filename
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[reporting] Model comparison forest plot saved to {outpath}")
    return outpath


# ─────────────────────────────────────────────────────────────────────────────
# FULL COX FOREST PLOT (all covariates from one model)
# ─────────────────────────────────────────────────────────────────────────────

def plot_full_cox_forest(
    cox_result: dict,
    model_name: str = "Fully Adjusted",
    output_filename: str = "forest_plot_full_cox.png",
) -> Path:
    """Forest plot for all covariates in a single Cox model."""
    ensure_output_dir()

    if "error" in cox_result or "summary" not in cox_result:
        return None

    summary = cox_result["summary"].copy()
    summary = summary.sort_values("exp(coef)")

    labels = summary.index.tolist()
    hrs = summary["exp(coef)"].values
    lower = summary["exp(coef) lower 95%"].values
    upper = summary["exp(coef) upper 95%"].values
    pvals = summary["p"].values

    fig, ax = plt.subplots(figsize=(12, max(4, len(labels) * 0.45 + 2)))

    y_pos = np.arange(len(labels))

    for i in range(len(labels)):
        is_treatment = "treatment" in labels[i].lower()
        color = "#B2182B" if is_treatment else ("#2166AC" if pvals[i] < 0.05 else "#999999")
        lw = 3 if is_treatment else 2
        ms = 10 if is_treatment else 7
        ax.plot([lower[i], upper[i]], [y_pos[i]] * 2, color=color, linewidth=lw)
        ax.plot(hrs[i], y_pos[i], "D" if is_treatment else "o",
                color=color, markersize=ms, zorder=5)

    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_yticks(y_pos)

    # Clean up labels
    clean_labels = []
    for label in labels:
        label = label.replace("treatment_continuation", "Continuation vs Fixed-Duration")
        label = label.replace("_", " ").replace("  ", " ")
        clean_labels.append(label)

    ax.set_yticklabels(clean_labels, fontsize=9)
    ax.set_xlabel("Hazard Ratio (95% CI)", fontsize=12, fontweight="bold")
    ax.set_title(f"{model_name} Cox Model — Hazard Ratios\nwith 95% Confidence Intervals",
                 fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")

    for i in range(len(labels)):
        text = f"{hrs[i]:.2f} ({lower[i]:.2f}–{upper[i]:.2f}) p={pvals[i]:.3f}"
        ax.text(ax.get_xlim()[1] * 1.02, y_pos[i], text,
                fontsize=7, va="center", fontfamily="monospace")

    plt.tight_layout()
    outpath = OUTPUT_DIR / output_filename
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[reporting] Full Cox forest plot saved to {outpath}")
    return outpath


# ─────────────────────────────────────────────────────────────────────────────
# SUBGROUP FOREST PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_subgroup_forest(
    subgroup_df,
    output_filename: str = "forest_subgroup_analysis.png",
    title: str = "Treatment HR by Covariate Category\n(Continuation vs Fixed-Duration)",
) -> Path:
    """Forest plot for subgroup analysis results."""
    ensure_output_dir()

    if subgroup_df.empty:
        return None

    fig_height = max(11, len(subgroup_df) * 1.0 + 4.0)
    fig, ax = plt.subplots(figsize=(20, fig_height))

    y_pos = np.arange(len(subgroup_df))

    lower_vals = subgroup_df["HR_lower"].astype(float).values
    upper_vals = subgroup_df["HR_upper"].astype(float).values
    positive_lower = lower_vals[lower_vals > 0]
    x_min = max(0.35, positive_lower.min() * 0.85) if len(positive_lower) else 0.5
    x_max = max(2.2, upper_vals.max() * 1.20) if len(upper_vals) else 2.0
    ax.set_xlim(x_min, x_max)

    for i in range(len(subgroup_df)):
        shade = "#F7F9FC" if i % 2 == 0 else "#FFFFFF"
        ax.axhspan(i - 0.5, i + 0.5, color=shade, zorder=0)

    for i, (_, row) in enumerate(subgroup_df.iterrows()):
        is_overall = row.get("Variable") == "Overall"
        color = "#B2182B" if is_overall else "#2166AC"
        lw = 3 if is_overall else 2
        ms = 10 if is_overall else 7
        marker = "D" if is_overall else "o"

        ax.plot([row["HR_lower"], row["HR_upper"]], [y_pos[i]] * 2,
                color=color, linewidth=lw)
        ax.plot(row["HR"], y_pos[i], marker,
                color=color, markersize=ms, zorder=5)

    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_yticks(y_pos)

    variable_display = {
        "age_group": "Age Group",
        "race_group": "Race",
        "ecog_binary": "ECOG",
        "pdl1_cat": "PD-L1",
        "histology_cat": "Histology",
        "pembro_with_chemo": "Treatment Type",
    }
    labels = []
    for _, row in subgroup_df.iterrows():
        if row["Variable"] == "Overall":
            labels.append("Overall")
        else:
            var_label = variable_display.get(row["Variable"], row["Variable"])
            labels.append(f"  {var_label}: {row['Level']}")
    ax.set_yticklabels(labels, fontsize=14, fontweight="bold")

    ax.set_xlabel("Hazard Ratio (95% CI)", fontsize=15, fontweight="bold")
    ax.set_title(title, fontsize=18, fontweight="bold", pad=16)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", length=0)

    ax.text(0.02, 1.02, "Favors Continuation", transform=ax.transAxes,
            fontsize=12, color="#2166AC", va="bottom", fontweight="bold")
    ax.text(0.98, 1.02, "Favors Fixed-Duration", transform=ax.transAxes,
            fontsize=12, color="#B2182B", ha="right", va="bottom", fontweight="bold")

    for i, (_, row) in enumerate(subgroup_df.iterrows()):
        text = f"{row['HR']:.2f} ({row['HR_lower']:.2f}–{row['HR_upper']:.2f})  n={int(row['N'])}"
        ax.text(
            1.02,
            y_pos[i],
            text,
            transform=ax.get_yaxis_transform(),
            fontsize=11,
            va="center",
            fontfamily="monospace",
            fontweight="bold" if row.get("Variable") == "Overall" else "normal",
        )

    fig.subplots_adjust(left=0.42, right=0.84, top=0.92, bottom=0.05)
    outpath = OUTPUT_DIR / output_filename
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[reporting] Subgroup forest plot saved to {outpath}")
    return outpath


# ─────────────────────────────────────────────────────────────────────────────
# SCHOENFELD RESIDUAL PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_schoenfeld_residuals(
    ph_result: dict,
    output_filename: str = "schoenfeld_residuals.png",
) -> Path:
    """
    Plot Schoenfeld residuals for key covariates.
    Uses the fitted CoxPH model from test_proportional_hazards.
    """
    ensure_output_dir()

    cph = ph_result.get("cph")
    model_df = ph_result.get("model_df")
    if cph is None or model_df is None:
        return None

    # Get Schoenfeld residuals
    try:
        schoenfeld = cph.compute_residuals(model_df, kind="schoenfeld")
    except Exception as e:
        print(f"[reporting] Could not compute Schoenfeld residuals: {e}")
        return None

    # Select up to 6 covariates to plot
    cols_to_plot = schoenfeld.columns[:min(6, len(schoenfeld.columns))]
    n_plots = len(cols_to_plot)

    if n_plots == 0:
        return None

    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, col in enumerate(cols_to_plot):
        ax = axes[idx]
        residuals = schoenfeld[col].dropna()
        times = residuals.index

        ax.scatter(times, residuals.values, alpha=0.3, s=15, color="#2166AC")

        # Add LOWESS smoothing
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smooth = lowess(residuals.values, times.values, frac=0.6)
            ax.plot(smooth[:, 0], smooth[:, 1], color="#B2182B",
                    linewidth=2, label="LOWESS")
        except ImportError:
            # Simple rolling mean fallback
            sorted_idx = np.argsort(times)
            window = max(10, len(times) // 10)
            rolling = pd.Series(residuals.values[sorted_idx]).rolling(
                window, center=True).mean()
            ax.plot(times.values[sorted_idx], rolling.values,
                    color="#B2182B", linewidth=2, label="Rolling mean")

        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        clean_name = col.replace("treatment_continuation",
                                 "Continuation vs FD").replace("_", " ")
        ax.set_title(clean_name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Time (months)", fontsize=9)
        ax.set_ylabel("Schoenfeld Residual", fontsize=9)
        ax.legend(fontsize=8)

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Schoenfeld Residuals — PH Assumption Check",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    outpath = OUTPUT_DIR / output_filename
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[reporting] Schoenfeld residual plots saved to {outpath}")
    return outpath


# ─────────────────────────────────────────────────────────────────────────────
# LANDMARK SENSITIVITY PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_landmark_sensitivity(
    landmark_df,
    output_filename: str = "landmark_sensitivity.png",
) -> Path:
    """Plot treatment HR across different landmark times."""
    ensure_output_dir()

    if landmark_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))

    x = landmark_df["Landmark (months)"].values
    hr = landmark_df["HR"].values
    lower = landmark_df["HR_lower"].values
    upper = landmark_df["HR_upper"].values

    ax.fill_between(x, lower, upper, alpha=0.2, color="#2166AC")
    ax.plot(x, hr, "D-", color="#2166AC", linewidth=2, markersize=10)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    for i in range(len(x)):
        ax.annotate(
            f"HR={hr[i]:.3f}\n({lower[i]:.3f}-{upper[i]:.3f})",
            xy=(x[i], hr[i]),
            xytext=(0, 20),
            textcoords="offset points",
            fontsize=8,
            ha="center",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.8),
        )

    ax.set_xlabel("Landmark Time (months from pembrolizumab start)",
                  fontsize=12, fontweight="bold")
    ax.set_ylabel("Hazard Ratio (Continuation vs Fixed-Duration)",
                  fontsize=12, fontweight="bold")
    ax.set_title("Landmark Sensitivity Analysis\nTreatment HR at Different Landmark Times",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    outpath = OUTPUT_DIR / output_filename
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[reporting] Landmark sensitivity plot saved to {outpath}")
    return outpath


# ─────────────────────────────────────────────────────────────────────────────
# SAVE COX TABLES
# ─────────────────────────────────────────────────────────────────────────────

def save_cox_tables(
    comparison_df, subgroup_df, ph_result, landmark_df,
    cox_results, lasso_result=None,
) -> None:
    """Save all Cox-related tables as CSVs."""
    ensure_output_dir()

    comparison_df.to_csv(OUTPUT_DIR / "cox_model_comparison.csv",
                         index=False, encoding="utf-8-sig")
    print(f"[reporting] Model comparison saved to {OUTPUT_DIR / 'cox_model_comparison.csv'}")

    if not subgroup_df.empty:
        subgroup_df.to_csv(OUTPUT_DIR / "cox_subgroup_analysis.csv",
                           index=False, encoding="utf-8-sig")
        print(f"[reporting] Subgroup analysis saved to {OUTPUT_DIR / 'cox_subgroup_analysis.csv'}")

    if ph_result and not ph_result.get("test_results", pd.DataFrame()).empty:
        ph_result["test_results"].to_csv(
            OUTPUT_DIR / "cox_ph_test_schoenfeld.csv", encoding="utf-8-sig")
        print(f"[reporting] PH test saved to {OUTPUT_DIR / 'cox_ph_test_schoenfeld.csv'}")

    if not landmark_df.empty:
        landmark_df.to_csv(OUTPUT_DIR / "cox_landmark_sensitivity.csv",
                           index=False, encoding="utf-8-sig")
        print(f"[reporting] Landmark sensitivity saved to {OUTPUT_DIR / 'cox_landmark_sensitivity.csv'}")

    for model_name, result in cox_results.items():
        if "summary" in result:
            result["summary"].to_csv(
                OUTPUT_DIR / f"cox_{model_name}_model_summary.csv",
                encoding="utf-8-sig")
            print(f"[reporting] {model_name} model summary saved")

    if lasso_result and "summary" in lasso_result:
        lasso_result["summary"].to_csv(
            OUTPUT_DIR / "cox_lasso_model_summary.csv", encoding="utf-8-sig")
        print(f"[reporting] LASSO model summary saved")
