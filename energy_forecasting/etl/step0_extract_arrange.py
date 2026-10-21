"""
STEP 0 — Extract from Database
================================
Pulls raw consumption readings from the database for the configured meter.

Input : MySQL database (consumptions table)
Output: raw_data.csv (raw meter readings)
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import mysql.connector
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import EXTRACTION, RAW_INPUT, DB_CONFIG

TABLE = EXTRACTION["table"]
SN = EXTRACTION["sn"]
DATE_COL = EXTRACTION["date_column"]
DATE_UNIT = EXTRACTION["date_unit"]


def run():
    print("=" * 60)
    print("  STEP 0 — Extract from Database")
    print("=" * 60)

    # 1. Connect and query
    print(f"\n  Connecting to database...")
    conn = mysql.connector.connect(**DB_CONFIG)

    query = f"""
        SELECT {DATE_COL}, sn, consumption_diff
        FROM {TABLE}
        WHERE sn = %s
        ORDER BY {DATE_COL} DESC
    """

    print(f"  Querying table '{TABLE}' for SN {SN}...")
    df = pd.read_sql(query, conn, params=(SN,))
    conn.close()
    print(f"  Rows fetched: {len(df):,}")

    # 2. Parse date
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], unit=DATE_UNIT, errors="coerce")
    print(f"  Date range: {df[DATE_COL].min()} → {df[DATE_COL].max()}")

    # 3. Sort chronologically
    df = df.sort_values(DATE_COL, ascending=False).reset_index(drop=True)
    print(f"\n  Rows: {len(df):,}")
    print(df.head(3).to_string())

    # 4. Save
    df.to_csv(RAW_INPUT, index=False)
    print(f"\n  Saved → {RAW_INPUT}")
    print(f"  Shape: {df.shape}")

    return df


if __name__ == "__main__":
    run()