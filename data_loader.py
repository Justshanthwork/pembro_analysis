"""
data_loader.py  —  Snowflake → Parquet cache → Analysis
=========================================================
Flow on first run:
    1. Connect to Snowflake (browser SSO)
    2. Pull each table with push-down filters  (only rows/columns needed)
    3. Save to local .parquet files in CACHE_DIR

Flow on subsequent runs (fast):
    1. Load from .parquet — no Snowflake connection needed
    2. Skip tables whose cache is still fresh (< CACHE_MAX_AGE_DAYS)

Force a full re-pull:
    python main.py --refresh

Force re-pull of specific tables:
    python main.py --refresh demographics dose

Delete cache manually:
    del cache\\demographics.parquet          (Windows)
    rm cache/demographics.parquet           (Mac/Linux)
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from config import (
    SNOWFLAKE_CONFIG, CACHE_DIR, CACHE_MAX_AGE_DAYS,
    DIAGNOSIS_START, DIAGNOSIS_END,
    CHEMO_AGENTS,
)

# ─────────────────────────────────────────────────────────────────────────────
# Snowflake table name mapping  (key → full table name in Snowflake)
# ─────────────────────────────────────────────────────────────────────────────
_PFX   = "IC_PRECISIONQ_STN_DEATH_NSCLC_"
_SFX   = "_20260310_190335"
_DB    = SNOWFLAKE_CONFIG["database"]
_SCH   = SNOWFLAKE_CONFIG["schema"]

SNOWFLAKE_TABLES = {
    k: f'"{_DB}"."{_SCH}"."{_PFX}{k.upper()}{_SFX}"'
    for k in [
        "demographics", "disease", "lot", "dose", "labs", "vitals",
        "biomarker", "staginghistory", "lotresponse", "metastases",
        "comorbidities", "procedure", "biopsy", "riskscores",
        "diseasehistory", "medicalcondition", "drugmodifications",
        "visithistory", "futurevisits", "cohort", "patientcuration",
        "regimen", "rsi", "controltable", "division", "remappedmpiid",
    ]
}

# ─────────────────────────────────────────────────────────────────────────────
# Per-table push-down WHERE clauses
# Each filter reduces Snowflake I/O significantly before the data hits Python.
# ─────────────────────────────────────────────────────────────────────────────
_chemo_list = ", ".join(f"'{a}'" for a in CHEMO_AGENTS)
_icd_prefixes = ("'E10'", "'E11'", "'E13'", "'E14'",
                 "'J43'", "'J44'", "'J45'",
                 "'I21'", "'I22'", "'I50'", "'I63'", "'I64'", "'I65'", "'I73'",
                 "'N18'", "'N19'")
_med_pattern = (
    r".*(beta.blocker|diuretic|painkiller|analgesic|opioid"
    r"|corticosteroid|dexamethasone|prednisone|steroid"
    r"|renin.angiotensin|ace inhibitor|angiotensin|rasi"
    r"|anticoagulant|warfarin|heparin|enoxaparin).*"
)

TABLE_FILTERS: dict[str, str] = {
    # Only adults with NSCLC diagnosis in study window
    "demographics": "AGE_DX >= 18",

    # Only metastatic C34.X diagnosed in study window
    "disease": (
        f"UPPER(CANCER_CODE) LIKE 'C34%' "
        f"AND DIAG_DATE BETWEEN '{DIAGNOSIS_START}' AND '{DIAGNOSIS_END}'"
    ),

    # Only first-line metastatic LOT
    "lot": (
        "CAST(LOT AS VARCHAR) = '1' "
        "AND CAST(METASTATIC AS VARCHAR) IN ('1', '1.0')"
    ),

    # Only pembrolizumab infusions + chemo agents we care about
    "dose": (
        "LOWER(GENERIC_NAME) LIKE '%pembrolizumab%' "
        "OR LOWER(BRAND_NAME) LIKE '%keytruda%' "
        f"OR LOWER(GENERIC_NAME) IN ({_chemo_list})"
    ),

    # Only PD-L1 biomarker results
    "biomarker": "UPPER(BIOMARKER_NAME) LIKE '%PD-L1%'",

    # Only ECOG measurements — confirmed working via vitals
    "vitals": "UPPER(TEST_NAME) LIKE '%ECOG%'",
    "labs":   None,   # pull all — column name unknown, will discover on next pull

    # Only brain metastases
    "metastases": "LOWER(METASTATIC_SITE) LIKE '%brain%'",

    # Only comorbidities with relevant ICD codes
    "comorbidities": (
        f"LEFT(UPPER(ICD_CODE), 3) IN ({', '.join(_icd_prefixes)})"
    ),

    # Pull all — RLIKE pattern may not match Snowflake terminology
    "medicalcondition": None,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-convert columns that look like dates."""
    date_hints = {
        "date", "diag_date", "staging_date", "metastatic_date",
        "start_date", "end_date", "last_activity", "test_date",
        "result_date", "specimen_collection_date", "event_date",
        "change_date", "visit_date", "lot_end_date",
        "drug_exposure_start_date", "drug_exposure_end_date",
        "cond_st_date", "cond_end_date", "date_event", "date_end",
        "disease_date", "last_curation_date", "remapped_date",
        "change_date_end", "date_death",
    }
    for col in df.columns:
        cl = col.lower()
        if cl in date_hints or cl.endswith("_date") or cl.startswith("date_"):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass
    return df


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.parquet"


def _cache_is_fresh(name: str) -> bool:
    """Return True if cache file exists and is not older than CACHE_MAX_AGE_DAYS."""
    p = _cache_path(name)
    if not p.exists():
        return False
    if CACHE_MAX_AGE_DAYS <= 0:
        return True
    age_days = (datetime.now(timezone.utc).timestamp() - p.stat().st_mtime) / 86400
    return age_days < CACHE_MAX_AGE_DAYS


def _save_cache(name: str, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_cache_path(name), index=False)


def _load_cache(name: str) -> pd.DataFrame:
    df = pd.read_parquet(_cache_path(name))
    df = _parse_dates(df)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Snowflake pull
# ─────────────────────────────────────────────────────────────────────────────
def _pull_from_snowflake(table_names: list[str]) -> dict[str, pd.DataFrame]:
    """Open one Snowflake connection and pull all requested tables."""
    try:
        import snowflake.connector
    except ImportError:
        raise ImportError(
            "snowflake-connector-python not installed.\n"
            "Run:  pip install snowflake-connector-python pyarrow"
        )

    print("[data_loader] Opening Snowflake connection (browser SSO will open)...")
    con = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
    results: dict[str, pd.DataFrame] = {}

    try:
        for name in table_names:
            table = SNOWFLAKE_TABLES.get(name)
            if table is None:
                print(f"  ✗ {name:20s}  No table mapping — skipped")
                continue

            where = TABLE_FILTERS.get(name)
            sql = f"SELECT * FROM {table}"
            if where:  # None means no filter — pull all rows
                sql += f"\nWHERE {where}"

            print(f"  ↓ {name:20s}  querying...", end="", flush=True)
            try:
                df = pd.read_sql(sql, con)
                # Snowflake returns UPPER column names — normalise
                df.columns = df.columns.str.lower().str.strip()
                df = _parse_dates(df)
                _save_cache(name, df)
                results[name] = df
                print(f"\r  ✓ {name:20s}  {len(df):>8,} rows  [cached → {name}.parquet]")
            except Exception as e:
                print(f"\r  ✗ {name:20s}  query failed: {e}")
    finally:
        con.close()
        print("[data_loader] Snowflake connection closed.")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def load_tables(
    table_names: list[str] | None = None,
    force_refresh: bool | list[str] = False,
) -> dict[str, pd.DataFrame]:
    """
    Load tables from local parquet cache, pulling from Snowflake as needed.

    Parameters
    ----------
    table_names : list of str, optional
        Which tables to load. Defaults to the 11 tables used by the analysis.
    force_refresh : bool or list of str
        True  → re-pull ALL tables from Snowflake.
        list  → re-pull only those table names (e.g. ["dose", "demographics"]).
        False → use cache when fresh.
    """
    if table_names is None:
        table_names = [
            "demographics", "disease", "lot", "dose",
            "biomarker", "vitals", "labs",
            "metastases", "comorbidities", "medicalcondition",
        ]

    # Determine which tables need a Snowflake pull
    if force_refresh is True:
        to_pull = list(table_names)
    elif isinstance(force_refresh, list):
        to_pull = [t for t in table_names if t in force_refresh]
    else:
        to_pull = [t for t in table_names if not _cache_is_fresh(t)]

    # Pull from Snowflake if needed
    if to_pull:
        print(f"[data_loader] Pulling {len(to_pull)} table(s) from Snowflake: {to_pull}")
        _pull_from_snowflake(to_pull)
    else:
        print(f"[data_loader] All {len(table_names)} tables loaded from cache (no Snowflake needed)")

    # Load everything from cache
    tables: dict[str, pd.DataFrame] = {}
    for name in table_names:
        p = _cache_path(name)
        if p.exists():
            df = _load_cache(name)
            tables[name] = df
            src = "cache" if name not in to_pull else "Snowflake→cache"
            print(f"  ✓ {name:20s}  {len(df):>8,} rows  [{src}]")
        else:
            print(f"  ✗ {name:20s}  not available (Snowflake pull may have failed)")

    return tables


def cache_status() -> None:
    """Print the age and row count of each cached table."""
    print("\nCache status:")
    print(f"  Directory: {CACHE_DIR}")
    print(f"  {'Table':<22} {'Rows':>8}  {'Age':>10}  {'Status'}")
    print(f"  {'-'*22} {'-'*8}  {'-'*10}  {'-'*10}")
    now = datetime.now(timezone.utc).timestamp()
    for name, _ in SNOWFLAKE_TABLES.items():
        p = _cache_path(name)
        if p.exists():
            age_h = (now - p.stat().st_mtime) / 3600
            age_str = f"{age_h:.1f}h ago" if age_h < 48 else f"{age_h/24:.1f}d ago"
            try:
                rows = len(pd.read_parquet(p, columns=["mpi_id"]))
            except Exception:
                rows = -1
            fresh = "OK" if _cache_is_fresh(name) else "STALE"
            print(f"  {name:<22} {rows:>8,}  {age_str:>10}  {fresh}")
        else:
            print(f"  {name:<22} {'—':>8}  {'—':>10}  MISSING")
    print()
