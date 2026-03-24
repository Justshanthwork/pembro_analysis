"""
synthetic_data.py — Generate realistic synthetic IC PrecisionQ data for pipeline testing
========================================================================================
Creates fake-but-structurally-valid data that mirrors the real schema.
The synthetic cohort encodes a known survival difference between fixed-duration
and continuation arms so you can verify the statistical pipeline produces
sensible output.

Usage:
    from synthetic_data import generate_all_synthetic_tables
    tables = generate_all_synthetic_tables()   # dict of DataFrames
"""

import numpy as np
import pandas as pd
from datetime import timedelta
from config import (
    SYNTHETIC_N_PATIENTS, SYNTHETIC_SEED,
    DIAGNOSIS_START, DIAGNOSIS_END, FOLLOWUP_CUTOFF,
)


def generate_all_synthetic_tables(
    n: int = SYNTHETIC_N_PATIENTS,
    seed: int = SYNTHETIC_SEED,
) -> dict[str, pd.DataFrame]:
    """Return dict of table_name -> DataFrame matching IC PrecisionQ schema."""
    rng = np.random.default_rng(seed)

    # ── patient-level backbone ──────────────────────────────────────────
    mpi_ids = np.arange(1, n + 1)

    # Demographics
    gender = rng.choice(["M", "F"], size=n, p=[0.55, 0.45])
    age_dx = rng.integers(45, 85, size=n)
    race = rng.choice(
        ["White", "Black", "Asian", "Other", "Unknown"],
        size=n, p=[0.65, 0.15, 0.08, 0.07, 0.05],
    )
    ethnicity = rng.choice(
        ["Not Hispanic or Latino", "Hispanic or Latino", "Unknown"],
        size=n, p=[0.82, 0.12, 0.06],
    )
    payer = rng.choice(
        ["Medicare", "Commercial", "Medicaid", "Other", "Unknown"],
        size=n, p=[0.45, 0.30, 0.10, 0.10, 0.05],
    )
    smoking = rng.choice(
        ["Current", "Former", "Never", "Unknown"],
        size=n, p=[0.25, 0.45, 0.20, 0.10],
    )

    # Diagnosis dates (uniform within window)
    dx_start = pd.Timestamp(DIAGNOSIS_START)
    dx_end = pd.Timestamp(DIAGNOSIS_END)
    dx_range_days = (dx_end - dx_start).days
    diag_dates = pd.to_datetime([
        dx_start + timedelta(days=int(d))
        for d in rng.integers(0, dx_range_days, size=n)
    ])

    # Histology
    histology = rng.choice(
        ["Adenocarcinoma", "Squamous Cell Carcinoma", "Large Cell",
         "NSCLC NOS", "Adenosquamous"],
        size=n, p=[0.50, 0.30, 0.05, 0.10, 0.05],
    )

    # De novo vs recurrent metastatic
    de_novo = rng.choice([1, 0], size=n, p=[0.70, 0.30])

    # Metastatic date: de novo = diag_date; recurrent = diag_date + 3-24 months
    met_dates = []
    for i in range(n):
        if de_novo[i]:
            met_dates.append(diag_dates[i])
        else:
            offset = int(rng.integers(90, 730))
            met_dates.append(diag_dates[i] + timedelta(days=offset))
    met_dates = pd.to_datetime(met_dates)

    # ── treatment assignment ────────────────────────────────────────────
    # We simulate 3 groups to test the pipeline:
    #   Group A (~40%): Fixed-duration (pembro ~24 months, last infusion 22-26 mo)
    #   Group B (~30%): Continuation (pembro > 26 months)
    #   Group C (~30%): Early stoppers / excluded (pembro < 22 months or die early)
    group = rng.choice(["A", "B", "C"], size=n, p=[0.40, 0.30, 0.30])

    # LOT start = metastatic date + 0-30 days
    lot_start = pd.to_datetime([
        met_dates[i] + timedelta(days=int(rng.integers(0, 30)))
        for i in range(n)
    ])

    # Pembro with chemo?
    with_chemo = rng.choice([0, 1], size=n, p=[0.40, 0.60])

    # Treatment duration in days by group
    tx_duration_days = np.zeros(n, dtype=int)
    for i in range(n):
        if group[i] == "A":
            # ~22-26 months = 660-790 days
            tx_duration_days[i] = int(rng.integers(660, 791))
        elif group[i] == "B":
            # 27-42 months = 810-1260 days
            tx_duration_days[i] = int(rng.integers(810, 1261))
        else:
            # Early stop: 1-18 months
            tx_duration_days[i] = int(rng.integers(30, 540))

    lot_end = pd.to_datetime([
        lot_start[i] + timedelta(days=int(tx_duration_days[i]))
        for i in range(n)
    ])

    # ── survival outcomes ───────────────────────────────────────────────
    # Encode a known HR: continuation slightly better than fixed-duration
    # True HR ~ 0.75 for continuation vs fixed-duration
    followup_cutoff = pd.Timestamp(FOLLOWUP_CUTOFF)

    # Landmark date = lot_start + 29 months (~882 days)
    landmark_dates = pd.to_datetime([
        lot_start[i] + timedelta(days=882) for i in range(n)
    ])

    # Alive at landmark?
    alive_at_landmark = np.ones(n, dtype=bool)
    death_date = [pd.NaT] * n
    last_activity = [pd.NaT] * n

    for i in range(n):
        if group[i] == "C":
            # 50% die before landmark
            if rng.random() < 0.50:
                days_to_death = int(rng.integers(30, 850))
                death_date[i] = lot_start[i] + timedelta(days=days_to_death)
                if death_date[i] < landmark_dates[i]:
                    alive_at_landmark[i] = False
                last_activity[i] = death_date[i]
            else:
                # alive, censor at random point
                last_activity[i] = min(
                    lot_start[i] + timedelta(days=int(rng.integers(200, 1400))),
                    followup_cutoff,
                )
        elif group[i] == "A":
            # Fixed-duration: median post-landmark survival ~18 months
            # 60% die during follow-up
            if rng.random() < 0.60:
                post_lm_days = int(rng.exponential(scale=540))  # ~18 mo median
                death_date[i] = landmark_dates[i] + timedelta(days=max(1, post_lm_days))
                if death_date[i] > followup_cutoff:
                    death_date[i] = pd.NaT  # censored
                    last_activity[i] = followup_cutoff
                else:
                    last_activity[i] = death_date[i]
            else:
                last_activity[i] = followup_cutoff
        else:
            # Continuation: median post-landmark survival ~24 months (better)
            if rng.random() < 0.50:
                post_lm_days = int(rng.exponential(scale=720))  # ~24 mo median
                death_date[i] = landmark_dates[i] + timedelta(days=max(1, post_lm_days))
                if death_date[i] > followup_cutoff:
                    death_date[i] = pd.NaT
                    last_activity[i] = followup_cutoff
                else:
                    last_activity[i] = death_date[i]
            else:
                last_activity[i] = followup_cutoff

    death_date = pd.to_datetime(death_date)
    last_activity = pd.to_datetime(last_activity)

    # ── Build DataFrames ────────────────────────────────────────────────

    # DEMOGRAPHICS
    demographics = pd.DataFrame({
        "mpi_id": mpi_ids,
        "flag": 0,
        "division_mask": 1,
        "combined_div_mpi_id": [f"1_{mid}" for mid in mpi_ids],
        "gender": gender,
        "age_dx": age_dx,
        "age_tx": age_dx + rng.uniform(0, 1, size=n).round(1),
        "payer": payer,
        "race": race,
        "ethnicity": ethnicity,
        "smoking_history": smoking,
        "provider_mpi_id": rng.integers(1000, 2000, size=n).astype(float),
        "specialty": rng.choice(["Medical Oncology", "Hematology/Oncology"], size=n),
        "provider_zip": rng.integers(10000, 99999, size=n).astype(float),
        "rsi": None,
        "last_activity": last_activity,
        "date_death": death_date,  # NaT if alive, actual date if died
        "other_primary": 0,
    })

    # DISEASE
    cancer_codes = rng.choice(
        ["C34.1", "C34.2", "C34.3", "C34.9"],
        size=n, p=[0.30, 0.25, 0.20, 0.25],
    )
    disease = pd.DataFrame({
        "mpi_id": mpi_ids,
        "division_mask": 1,
        "combined_div_mpi_id": [f"1_{mid}" for mid in mpi_ids],
        "diag_date": diag_dates,
        "cancer_code": cancer_codes,
        "cancer_stage": np.where(de_novo, "IV", rng.choice(["I", "II", "III", "IV"], size=n)),
        "t_value": rng.choice(["T1", "T2", "T3", "T4", "TX"], size=n),
        "n_value": rng.choice(["N0", "N1", "N2", "N3", "NX"], size=n),
        "m_value": "M1",  # all metastatic
        "g_value": np.nan,
        "staging_date": diag_dates,
        "is_rai": 0,
        "date_of_progression": np.nan,
        "metastatic_date": met_dates,
        "metastatic_site": np.nan,
        "disease_subtype": np.nan,
        "location": rng.choice(["Right Upper", "Right Lower", "Left Upper", "Left Lower", "Main Bronchus"], size=n),
        "histology": histology,
        "crpc_diagnosis_date": pd.NaT,
        "mcrpc_date": pd.NaT,
        "rsi": None,
        "condition_concept_id": None,
    })

    # LOT (line of therapy) — just LOT 1 for the primary analysis
    regimen_names = []
    for i in range(n):
        if with_chemo[i]:
            chemo = rng.choice(["Carboplatin + Pemetrexed", "Carboplatin + Paclitaxel"])
            regimen_names.append(f"Pembrolizumab + {chemo}")
        else:
            regimen_names.append("Pembrolizumab")

    lot_df = pd.DataFrame({
        "mpi_id": mpi_ids,
        "division_mask": 1,
        "combined_div_mpi_id": [f"1_{mid}" for mid in mpi_ids],
        "lot": 1,
        "regimen": regimen_names,
        "start_date": lot_start,
        "end_date": lot_end,
        "duration_days": tx_duration_days,
        "duration_months": (tx_duration_days / 30.44).astype(int),
        "metastatic": 1,
        "no_div_lot": 1,
    })

    # DOSE — generate pembro infusion records every 21 days
    dose_rows = []
    for i in range(n):
        current = lot_start[i]
        end = lot_end[i]
        while current <= end:
            dose_rows.append({
                "mpi_id": mpi_ids[i],
                "division_mask": 1,
                "combined_div_mpi_id": f"1_{mpi_ids[i]}",
                "generic_name": "pembrolizumab",
                "brand_name": "Keytruda",
                "jcode": "J9271",
                "ndc": None,
                "drug_exposure_start_date": current,
                "drug_exposure_end_date": current,
                "dose_value": 200.0,
                "dose_units": "mg",
                "days_supply": 1.0,
                "strength": 200.0,
                "dose_weight": np.nan,
                "therapy_form": np.nan,
                "source_location": "Infusion Center",
                "frequency": "Q3W",
                "quantity": 1.0,
                "quantity_per_day": np.nan,
                "days_since_last_admin": np.nan,
                "supportive_care": 0,
                "drug_class": "PD-1 Inhibitor",
                "therapy_route": "IV",
                "drug_rsi": None,
                "mpi_id_lot": f"{mpi_ids[i]}_1",
                "drug_concept_id": None,
                "route_concept_id": None,
            })
            # Next infusion: 21 days +/- 3 day jitter
            jitter = int(rng.integers(-3, 4))
            current = current + timedelta(days=21 + jitter)

        # If with_chemo, add chemo doses for first ~4-6 cycles
        if with_chemo[i]:
            chemo_drug = "carboplatin" if "Carboplatin" in regimen_names[i] else "pemetrexed"
            n_chemo_cycles = int(rng.integers(4, 7))
            current = lot_start[i]
            for cycle in range(n_chemo_cycles):
                dose_rows.append({
                    "mpi_id": mpi_ids[i],
                    "division_mask": 1,
                    "combined_div_mpi_id": f"1_{mpi_ids[i]}",
                    "generic_name": chemo_drug,
                    "brand_name": chemo_drug.title(),
                    "jcode": None,
                    "ndc": None,
                    "drug_exposure_start_date": current,
                    "drug_exposure_end_date": current,
                    "dose_value": rng.uniform(300, 600),
                    "dose_units": "mg",
                    "days_supply": 1.0,
                    "strength": np.nan,
                    "dose_weight": np.nan,
                    "therapy_form": np.nan,
                    "source_location": "Infusion Center",
                    "frequency": "Q3W",
                    "quantity": 1.0,
                    "quantity_per_day": np.nan,
                    "days_since_last_admin": np.nan,
                    "supportive_care": 0,
                    "drug_class": "Chemotherapy",
                    "therapy_route": "IV",
                    "drug_rsi": None,
                    "mpi_id_lot": f"{mpi_ids[i]}_1",
                    "drug_concept_id": None,
                    "route_concept_id": None,
                })
                current = current + timedelta(days=21)

    dose_df = pd.DataFrame(dose_rows)

    # BIOMARKER — PD-L1 status
    pdl1_results = rng.choice(
        ["Positive (TPS ≥50%)", "Positive (TPS 1-49%)", "Negative (TPS <1%)", "Unknown"],
        size=n, p=[0.30, 0.30, 0.25, 0.15],
    )
    biomarker = pd.DataFrame({
        "mpi_id": mpi_ids,
        "division_mask": 1,
        "combined_div_mpi_id": [f"1_{mid}" for mid in mpi_ids],
        "testing_company": rng.choice(["Dako", "Ventana", "Other"], size=n),
        "commercial_test_name": "PD-L1 IHC 22C3 pharmDx",
        "genomic_panel_tested": None,
        "test_category": "IHC",
        "test_date": diag_dates + pd.to_timedelta(rng.integers(0, 30, size=n), unit="D"),
        "result_date": diag_dates + pd.to_timedelta(rng.integers(7, 45, size=n), unit="D"),
        "biomarker_name": "PD-L1",
        "test_result": pdl1_results,
        "sample_type": "Tissue",
        "karyotype": np.nan,
        "specimen_collection_method": rng.choice(["Biopsy", "Resection"], size=n),
        "specimen_collection_date": diag_dates,
        "rsi": None,
    })

    # LABS — ECOG PS (primary source; matches real schema: value=float, value_string=str)
    ecog_numeric = rng.choice([0, 1, 2, 3], size=n, p=[0.25, 0.45, 0.20, 0.10]).astype(float)
    labs = pd.DataFrame({
        "mpi_id": mpi_ids,
        "division_mask": 1,
        "combined_div_mpi_id": [f"1_{mid}" for mid in mpi_ids],
        "test_name": "ECOG",
        "test_date": lot_start,
        "value": ecog_numeric,                             # float64, e.g. 0.0, 1.0, 2.0
        "unit_value": None,
        "rsi": None,
        "measurement_concept_id": None,
        "value_string": ecog_numeric.astype(int).astype(str),  # "0", "1", "2", "3"
    })

    # RISKSCORES — kept as fallback; ECOG also included here for testing fallback path
    ecog_str = ecog_numeric.astype(int).astype(str)
    riskscores = pd.DataFrame({
        "mpi_id": mpi_ids,
        "division_mask": 1,
        "combined_div_mpi_id": [f"1_{mid}" for mid in mpi_ids],
        "test_date": lot_start,
        "value": ecog_str,
        "risk_name": "ECOG",
        "rsi": None,
    })

    # METASTASES — brain mets for ~20% of patients
    met_rows = []
    brain_met_sites = ["Brain", "Bone", "Liver", "Adrenal", "Contralateral Lung"]
    for i in range(n):
        n_sites = int(rng.integers(1, 4))
        sites = rng.choice(brain_met_sites, size=n_sites, replace=False)
        for site in sites:
            met_rows.append({
                "mpi_id": mpi_ids[i],
                "division_mask": 1,
                "combined_div_mpi_id": f"1_{mpi_ids[i]}",
                "metastatic_date": met_dates[i],
                "metastatic_site": site,
                "secondary_malig_code": None,
                "rsi": None,
            })
    metastases = pd.DataFrame(met_rows)

    # REGIMEN reference table
    unique_regimens = list(set(regimen_names))
    regimen_ref = pd.DataFrame({
        "regimen": unique_regimens,
        "regimen_category": [
            "IO+Chemo" if "+" in r else "IO Mono" for r in unique_regimens
        ],
        "regimen_group": "Pembrolizumab-based",
        "regimen_subgroup": [
            "Nonsquamous" if "Pemetrexed" in r
            else "Squamous" if "Paclitaxel" in r
            else "Monotherapy"
            for r in unique_regimens
        ],
    })

    # ── Assemble all tables ─────────────────────────────────────────────
    tables = {
        "demographics": demographics,
        "disease": disease,
        "lot": lot_df,
        "dose": dose_df,
        "biomarker": biomarker,
        "labs": labs,
        "riskscores": riskscores,
        "metastases": metastases,
        "regimen": regimen_ref,
    }

    return tables


if __name__ == "__main__":
    tables = generate_all_synthetic_tables()
    for name, df in tables.items():
        print(f"{name:20s}  {df.shape[0]:>7,} rows × {df.shape[1]} cols")
