"""
Database seeding script to populate MySQL dailyconsumptions table from source datasets.
Run from host: python backend/seed_db.py
"""
import os
import sys
from pathlib import Path
from urllib.parse import quote

from dotenv import dotenv_values

import pandas as pd
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENERGY_CSV = PROJECT_ROOT / "energy_forecasting" / "etl" / "data" / "generated dataset" / "raw_data_extended.csv"
WATER_CSV = PROJECT_ROOT / "water_forecasting" / "bdg2_office_water" / "water_consumption_office.csv"

def _database_url() -> str:
    """Resolve database URL from SEED_DATABASE_URL or .env configuration."""
    # MODIFY: set MYSQL_USER / MYSQL_PASSWORD / MYSQL_PORT / MYSQL_DATABASE in the root .env
    url = os.getenv("SEED_DATABASE_URL")
    if url:
        return url
    env = {**dotenv_values(PROJECT_ROOT / ".env"), **os.environ}
    pwd = env.get("MYSQL_PASSWORD", "")
    if not pwd or pwd.upper().startswith("CHANGE_ME"):
        sys.exit("MYSQL_PASSWORD is missing or still a CHANGE_ME placeholder in .env")
    return (
        f"mysql+pymysql://{quote(env.get('MYSQL_USER', 'admin'), safe='')}:{quote(pwd, safe='')}"
        f"@127.0.0.1:{env.get('MYSQL_PORT', '3307')}/{env.get('MYSQL_DATABASE', 'energy_mgmt')}"
    )


def seed_database():
    engine = create_engine(_database_url())

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sites (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255), client_id INT, address TEXT, project VARCHAR(100)
            )"""))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS dailyconsumptions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                date BIGINT, local_date DATE, total_consumption FLOAT, site_id INT
            )"""))
        if len(conn.execute(text("SELECT id FROM sites")).fetchall()) < 2:
            conn.execute(text("INSERT INTO sites (id, name, project) VALUES (1, 'Energy Site', 'energy') ON DUPLICATE KEY UPDATE name=name"))
            conn.execute(text("INSERT INTO sites (id, name, project) VALUES (2, 'Water Site', 'water') ON DUPLICATE KEY UPDATE name=name"))
        conn.execute(text("TRUNCATE TABLE dailyconsumptions"))

    # ── Energy: generated sub-minute meter deltas → hourly sum ────────────────
    print(f"Reading energy dataset: {ENERGY_CSV.name}")
    df_energy = pd.read_csv(ENERGY_CSV, usecols=["date", "consumption_diff"])
    df_energy["date"] = pd.to_datetime(df_energy["date"])
    df_energy_hourly = df_energy.set_index("date").resample("1h").sum()
    energy_last = df_energy_hourly.index.max()

    # ── Water: real BDG2 (litres, 2016-2017) → date-shift to align, keep litres ─
    print(f"Reading water dataset: {WATER_CSV.name}")
    df_water = pd.read_csv(WATER_CSV, usecols=["datetime", "consumption_l"])
    df_water["datetime"] = pd.to_datetime(df_water["datetime"])
    df_water = df_water.set_index("datetime").sort_index()
    shift = energy_last - df_water.index.max()
    df_water.index = df_water.index + shift
    print(f"  water date-shift: +{shift}  →  {df_water.index.min()} … {df_water.index.max()}")

    records = []
    for dt, row in df_energy_hourly.iterrows():
        records.append({"date": int(dt.timestamp() * 1000), "local_date": dt.date(),
                        "total_consumption": float(row["consumption_diff"]), "site_id": 1})
    for dt, row in df_water.iterrows():
        records.append({"date": int(dt.timestamp() * 1000), "local_date": dt.date(),
                        "total_consumption": float(row["consumption_l"]), "site_id": 2})

    pd.DataFrame(records).to_sql("dailyconsumptions", con=engine, if_exists="append", index=False)
    n_e = sum(r["site_id"] == 1 for r in records)
    print(f"Inserted {n_e} energy rows + {len(records) - n_e} water rows (litres).")


if __name__ == "__main__":
    seed_database()
