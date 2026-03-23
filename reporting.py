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
        "ECOG PS": "ecog_ps",
        "PD-L1 Status": "pdl1_status",
        "Histology": "histology",
        "De Novo vs Recurrent": "de_novo_vs_recurrent",
        "Brain Metastases": "brain_mets",
        "Pembro ± Chemo": "pembro_with_chemo",
    }

    for label, col in cat_vars.items():
        if col not in cohort_df.columns:
            continue
        all_rows.append({"Variable": label, **{g: "" for g in groups}})
        all_rows.extend(_summarize_categorical(cohort_df, col, group_col).to_dict("records"))

    table1 = pd.DataFrame(all_rows)

    # Save
    table1.to_csv(OUTPUT_DIR / "table1_baseline_characteristics.csv", index=False)
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

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot each KM curve
    for grp_name, kmf in sorted(km_results.items()):
        color = colors.get(grp_name, "gray")
        kmf.plot_survival_function(
            ax=ax,
            ci_show=True,
            color=color,
            linewidth=2,
            ci_alpha=0.15,
        )

    ax.set_xlabel("Time from Landmark (months)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Overall Survival Probability", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    ax.legend(fontsize=11, loc="lower left", frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add log-rank p-value
    if lr is not None:
        p_text = f"Log-rank p = {lr.p_value:.4f}" if lr.p_value >= 0.0001 else "Log-rank p < 0.0001"
        ax.text(
            0.98, 0.95, p_text,
            transform=ax.transAxes, fontsize=11,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
        )

    # Add median OS annotations
    summary = km_output["summary"]
    y_pos = 0.85
    for _, row in summary.iterrows():
        text = f'{row["Cohort"]}: Median {row["Median OS (months)"]:.1f} mo (95% CI: {row["95% CI Lower"]:.1f}–{row["95% CI Upper"]:.1f})'
        ax.text(
            0.98, y_pos, text,
            transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right",
            fontfamily="monospace",
        )
        y_pos -= 0.05

    # Number at risk table
    time_points = np.arange(0, ax.get_xlim()[1], 6)
    risk_data = {}
    for grp_name, kmf in sorted(km_results.items()):
        n_at_risk = []
        for t in time_points:
            nar = (kmf.durations >= t).sum() if hasattr(kmf, "durations") else 0
            # Better approach: use the survival table
            try:
                st = kmf.event_table
                nar = st.loc[st.index <= t, "at_risk"].iloc[-1] if len(st[st.index <= t]) > 0 else 0
            except Exception:
                pass
            n_at_risk.append(int(nar))
        risk_data[grp_name] = n_at_risk

    # Add number-at-risk below the plot
    ax2 = fig.add_axes([0.125, 0.02, 0.775, 0.08])
    ax2.set_xlim(ax.get_xlim())
    ax2.set_ylim(-0.5, len(risk_data) - 0.5)
    ax2.axis("off")

    for idx, (grp_name, counts) in enumerate(sorted(risk_data.items())):
        color = colors.get(grp_name, "gray")
        ax2.text(-0.5, idx, grp_name, fontsize=9, fontweight="bold",
                 ha="right", va="center", color=color, transform=ax2.transData)
        for j, t in enumerate(time_points):
            ax2.text(t, idx, str(counts[j]), fontsize=8,
                     ha="center", va="center", color=color)

    ax2.text(-0.5, len(risk_data) - 0.1 + 0.5, "No. at risk", fontsize=9,
             fontweight="bold", ha="right", va="center")

    plt.subplots_adjust(bottom=0.15)

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

def save_cohort_csv(cohort_df: pd.DataFrame, filename: str = "analysis_cohort.csv") -> Path:
    """Save the analysis-ready cohort to CSV."""
    ensure_output_dir()
    outpath = OUTPUT_DIR / filename
    cohort_df.to_csv(outpath, index=False)
    print(f"[reporting] Analysis cohort saved to {outpath}")
    return outpath
