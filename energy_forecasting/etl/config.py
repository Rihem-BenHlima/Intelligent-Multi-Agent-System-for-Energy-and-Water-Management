"""
config.py — Pipeline Configuration
===================================
Configuration settings, database parameters, schedules,
and file paths for the building energy ETL pipeline.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

# --- Directories -----------------------------------------------------------
BASE_DIR      = Path(__file__).resolve().parent
DATA_DIR      = BASE_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PLOTS_DIR     = BASE_DIR / "plots"
REPORTS_DIR   = BASE_DIR / "reports"

for d in [RAW_DIR, PROCESSED_DIR, PLOTS_DIR, REPORTS_DIR]:
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Directory creation is optional when mounted in read-only environments
        pass

# --- Database credentials --------------------------------------------------
# Loaded from root .env file
load_dotenv(BASE_DIR.parents[1] / ".env")

DB_CONFIG = {
    "host": os.environ["DB_HOST"],
    "user": os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"],
    "database": os.environ["DB_NAME"],
}

# --- Domain Constants -------------------------------------------------------
TIMEZONE   = "Africa/Tunis"
LATITUDE   = 36.8065
LONGITUDE  = 10.1815

# Work schedule: Mon–Fri, 08:00–18:00, excluding public holidays
WORK_START = 8
WORK_END   = 18
WEEKEND    = [5, 6]  # Saturday, Sunday

# Tunisian public holidays
TUNISIA_HOLIDAYS = pd.to_datetime([
    "2024-10-15", "2024-12-17",
    "2025-01-01", "2025-03-20", "2025-04-09", "2025-05-01",
    "2025-06-06", "2025-06-07", "2025-06-08",
    "2025-07-25", "2025-08-13", "2025-10-15", "2025-12-17",
    "2026-01-01", "2026-03-20", "2026-04-09", "2026-05-01",
    "2026-08-25",
])
HOLIDAY_SET = set(TUNISIA_HOLIDAYS.normalize())

# Seasons
SEASONS = {
    "summer": [6, 7, 8],
    "winter": [12, 1, 2],
    "spring": [3, 4, 5],
    "autumn": [9, 10, 11],
}

# --- Step 1: Clean & Aggregate -----------------------------------------------
STEP1_CFG = {
    "spike_quantile": 0.9995,
    "coverage_threshold": 0.35,
    "min_readings": 20,
}

# --- Step 2: Detect & Impute -------------------------------------------------
STEP2_CFG = {
    "outlier_iqr_k": 3.5,
    "zero_streak_hours": 4,
    "tier1_limit": 3,
    "tier2_window_days": 14,
    "tier2_min_cands": 3,
    "tier3_min_cands": 5,
    "noise_factor": 0.8,
    "smooth_sigma": 0.8,
}

# --- Step 3: Smooth & Validate -----------------------------------------------
STEP3_CFG = {
    "z_threshold": 3.5,
    "rolling_window": 168,
    "min_periods": 24,
    "profile_z": 3.0,
    "profile_noise": 0.15,
    "profile_sigma": 0.7,
}

# --- Database extraction ----------------------------------------------------
EXTRACTION = {
    "table": "consumptions",
    # Meter serial number used by step0 as the WHERE filter
    "sn": "YOUR_METER_SERIAL",
    "date_column": "date",
    "date_unit": "ms",
}

# --- File Paths --------------------------------------------------------------
RAW_INPUT    = RAW_DIR / "raw_data.csv"
STEP1_OUTPUT = PROCESSED_DIR / "step1_hourly.csv"
STEP2_OUTPUT = PROCESSED_DIR / "step2_imputed.csv"
STEP3_OUTPUT = PROCESSED_DIR / "step3_final.csv"