"""
data_loader.py - Load real CSVs or fall back to synthetic data
==============================================================
"""

import pandas as pd
from pathlib import Path
from config import DATA_DIR, FILES, USE_SYNTHETIC_IF_MISSING


def _parse_dates(df):
    """Auto-convert columns that look like dates."""
    date_hints = [
        "date", "diag_date", "staging_date", "metastatic_date",
        "start_date", "end_date", "last_activity", "test_date",
        "result_date", "specimen_collection_date", "event_date",
        "change_date", "visit_date", "lot_end_date",
        "drug_exposure_start_date", "drug_exposure_end_date",
        "cond_st_date", "cond_end_date", "date_event", "date_end",
        "disease_date", "last_curation_date", "remapped_date",
        "change_date_end",
    ]
    for col in df.columns:
        if col.lower() in date_hints or col.lower().endswith("_date"):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass
    return df


def load_tables(table_names=None):
    """
    Load requested tables from CSV.
    Falls back to synthetic data if USE_SYNTHETIC_IF_MISSING is True.
    """
    if table_names is None:
        table_names = list(FILES.keys())

    data_path = Path(DATA_DIR)
    real_data_available = data_path.exists() and any(
        (data_path / FILES[t]).exists() for t in table_names if t in FILES
    )

    if real_data_available:
        print(f"[data_loader] Loading real data from {data_path}")
        tables = {}
        for name in table_names:
            fpath = data_path / FILES[name]
            if fpath.exists():
                try:
                    df = pd.read_csv(fpath, low_memory=False, encoding="utf-8")
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(fpath, low_memory=False, encoding="latin-1")
                    except Exception:
                        df = pd.read_csv(fpath, low_memory=False, encoding="latin-1",
                                         on_bad_lines="skip")
                df.columns = df.columns.str.lower().str.strip()
                df = _parse_dates(df)
                tables[name] = df
                n = len(df)
                print(f"  v {name:20s}  {n:>7,} rows")
            else:
                print(f"  x {name:20s}  FILE NOT FOUND: {fpath.name}")
        return tables

    elif USE_SYNTHETIC_IF_MISSING:
        print("[data_loader] Real CSVs not found - generating synthetic data")
        from synthetic_data import generate_all_synthetic_tables
        tables = generate_all_synthetic_tables()
        return {k: v for k, v in tables.items() if k in table_names}

    else:
        raise FileNotFoundError(
            f"Data directory {data_path} not found and USE_SYNTHETIC_IF_MISSING=False. "
            f"Set PEMBRO_DATA_DIR env variable or place CSVs in {data_path}"
        )
