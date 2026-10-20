"""Shared configuration — office water ETL."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

HOUSEHOLD_ID = "Panther_office_Larry"
RAW_INPUT = PROJECT_DIR / "bdg2_office_water" / "water_consumption_office.csv"

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DIR = BASE_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

STEP1_OUTPUT = OUTPUT_DIR / "step1_hourly_raw.csv"
STEP2_OUTPUT = OUTPUT_DIR / "step2_imputed.csv"
STEP3_OUTPUT = OUTPUT_DIR / f"{HOUSEHOLD_ID}_final_clean.csv"

SERIES_START = None
MAX_SYNTHETIC_FRACTION = 0.05

RATE_CAP_L_PER_H = 10000
ZERO_RUN_GAP_HOURS = 96

# Statistical spike detection (catches real-but-anomalous multi-hour events,
# e.g. a genuine multi-day demand spike, that RATE_CAP_L_PER_H is too high to
# ever catch). Flags an hour as a spike only if it is BOTH statistically
# extreme for its (day-of-week, hour) slot AND practically large in absolute
# terms — the absolute floor exists because near-zero-variance hours (deep
# night) produce huge z-scores from a single trivial pulse otherwise.
SPIKE_Z_THRESHOLD = 2.5
SPIKE_MIN_ABS_L_PER_H = 200

TIER1_MAX = 3
TIER2_MAX = 24
MAX_IMPUTE_GAP_HOURS = 14 * 24
NOISE_FACTOR = 0.15

SMOOTH_SIGMA = 0.8    # display-only, never applied to modeling output
SAMPLE_WINDOW_DAYS = 14
