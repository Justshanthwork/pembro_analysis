"""
diagnose.py — Data load diagnostics
Run from pembro_analysis/:
    python diagnose.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DIR, FILES

FOCUS_TABLES = ["labs", "medicalcondition"]

print("=" * 65)
print("DIAGNOSTIC: labs + medicalcondition data load")
print("=" * 65)
print(f"\nDATA_DIR: {DATA_DIR}")
print(f"  Exists: {Path(DATA_DIR).exists()}")

if not Path(DATA_DIR).exists():
    print("\n  !! DATA_DIR does not exist — check PEMBRO_DATA_DIR env var or config.py")
    sys.exit(1)

# List all CSV files actually present in DATA_DIR
all_csvs = sorted(Path(DATA_DIR).glob("*.csv"))
print(f"\nCSV files in DATA_DIR ({len(all_csvs)} total):")
for f in all_csvs:
    print(f"  {f.name}")

print("\n" + "=" * 65)

for table in FOCUS_TABLES:
    expected_filename = FILES.get(table)
    fpath = Path(DATA_DIR) / expected_filename

    print(f"\nTable: {table.upper()}")
    print(f"  Expected filename : {expected_filename}")
    print(f"  Full path         : {fpath}")
    print(f"  File exists       : {fpath.exists()}")

    if not fpath.exists():
        matches = [f for f in all_csvs if table.upper() in f.name.upper()]
        if matches:
            print(f"  Possible matches  : {[f.name for f in matches]}")
        else:
            print(f"  No filename match found in DATA_DIR")
        continue

    import pandas as pd

    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(fpath, low_memory=False, encoding=enc, nrows=5)
            print(f"  Encoding          : {enc} (OK)")
            break
        except Exception as e:
            print(f"  Encoding {enc} failed: {e}")
            df = None

    if df is None:
        print("  !! Could not read file with any encoding")
        continue

    try:
        df_full = pd.read_csv(fpath, low_memory=False, encoding=enc)
        print(f"  Row count         : {len(df_full):,}")
        print(f"  Column count      : {len(df_full.columns)}")
        print(f"  Columns           : {list(df_full.columns)}")
        print(f"  Null counts (top 5 cols):")
        for col in list(df_full.columns)[:5]:
            n_null = df_full[col].isna().sum()
            print(f"    {col}: {n_null:,} nulls / {len(df_full):,} rows")
        print(f"  First 3 rows:")
        print(df_full.head(3).to_string(index=False))
    except Exception as e:
        print(f"  !! Error reading full file: {e}")

print("\n" + "=" * 65)
print("Diagnostic complete. Paste the output above back to Claude.")
print("=" * 65)
