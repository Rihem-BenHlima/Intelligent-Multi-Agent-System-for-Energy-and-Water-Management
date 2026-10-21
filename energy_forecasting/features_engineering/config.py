"""
config.py
=========
Paths, constants, and calendar definitions for 24h building energy
feature engineering.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import pandas as pd

# ─────────────────────────────────────────────────────────────
# PATH RESOLUTION
# ─────────────────────────────────────────────────────────────
FEATURES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FEATURES_DIR.parent

# ─────────────────────────────────────────────────────────────
# DIRECTORIES
# ─────────────────────────────────────────────────────────────
ETL_DIR     = PROJECT_ROOT / "etl" / "data" / "processed"

OUTPUT_DIR  = FEATURES_DIR / "output"
PLOTS_DIR   = FEATURES_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# FILE PATHS
# ─────────────────────────────────────────────────────────────
INPUT_CSV     = ETL_DIR      / "step3_final.csv"
WEATHER_CSV   = FEATURES_DIR / "tunisia_weather.csv"
FORECAST_CSV  = FEATURES_DIR / "tunisia_weather_forecast_24h.csv"
FEATURES_CSV  = OUTPUT_DIR   / "features.csv"

# ─────────────────────────────────────────────────────────────
# DOMAIN — Tunisia
# ─────────────────────────────────────────────────────────────
LATITUDE   = 36.8065
LONGITUDE  = 10.1815
TIMEZONE   = "Africa/Tunis"
HORIZON    = 24   # hours ahead to forecast

# Summer months (peak AC load — hardest season to forecast)
SUMMER_MONTHS = {6, 7, 8, 9}

TUNISIA_HOLIDAYS = pd.to_datetime([
    "2024-10-15", "2024-12-17",
    "2025-01-01", "2025-03-20", "2025-04-09", "2025-05-01",
    "2025-06-06", "2025-06-07", "2025-06-08",
    "2025-07-25", "2025-08-13", "2025-10-15", "2025-12-17",
    "2026-01-01", "2026-03-20", "2026-04-09", "2026-05-01",
    "2026-08-25",
])
HOLIDAY_SET = set(TUNISIA_HOLIDAYS.normalize())

# Degree day thresholds (°C) for Cooling/Heating Demand
CDD_THRESHOLDS = [18, 21, 25, 28, 32]
HDD_THRESHOLDS = [12, 15]

# ─────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────
TRAIN_RATIO  = 0.80
RANDOM_SEED  = 42
LOOKBACK     = 168
BATCH_SIZE   = 64
MAX_EPOCHS   = 100
PATIENCE     = 15
LR           = 1e-3
WEIGHT_DECAY = 1e-3

# ─────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────
N_PLOT          = 500
COLOR_ACTUAL    = "#1565C0"
COLOR_PRED      = "#E53935"
BLUE            = "#1565C0"
RED             = "#E53935"

ASHRAE_CVRMSE_GOOD  = 20.0
ASHRAE_CVRMSE_LIMIT = 30.0
MONTHLY_R2_TARGET   = 0.65

# ─────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────
def validate_paths(raise_on_missing: bool = True) -> dict:
    checks = {
        "PROJECT_ROOT": PROJECT_ROOT,
        "ETL_DIR":      ETL_DIR,
        "INPUT_CSV":    INPUT_CSV,
    }

    print("\n  [Config] Path check:")
    all_ok = True
    for name, path in checks.items():
        ok = path.exists()
        print(f"    {'✅' if ok else '❌'}  {name}: {path}")
        if not ok:
            all_ok = False

    for name, path in [("WEATHER_CSV", WEATHER_CSV), ("FEATURES_CSV", FEATURES_CSV)]:
        ok = path.exists()
        icon = "✅" if ok else "⚠️ "
        note = "" if ok else " (will be created on first run)"
        print(f"    {icon}  {name}: {path}{note}")

    print(f"    ✅  OUTPUT_DIR : {OUTPUT_DIR}")
    print(f"    ✅  PLOTS_DIR  : {PLOTS_DIR}")

    if not all_ok and raise_on_missing:
        missing = [n for n, p in checks.items() if not p.exists()]
        raise FileNotFoundError(
            f"\nMissing required paths: {missing}\n"
            f"Set PROJECT_ROOT manually in config.py if your layout differs."
        )
    return {n: p.exists() for n, p in checks.items()}


if __name__ == "__main__":
    print("=" * 58)
    print("  config.py — directory & constant check")
    print("=" * 58)
    print(f"\n  PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"  FEATURES_DIR : {FEATURES_DIR}")
    print(f"  Holidays     : {len(TUNISIA_HOLIDAYS)} dates")
    print(f"  Summer months: {sorted(SUMMER_MONTHS)}")
    validate_paths(raise_on_missing=False)
    print("=" * 58)