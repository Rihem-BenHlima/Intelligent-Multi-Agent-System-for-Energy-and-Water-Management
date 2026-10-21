"""
config.py — Centralised path configuration
=============================================
All paths are derived from this file's own location (BASE_DIR), so every
script that does `import config` works regardless of the current working
directory it's launched from (e.g. `python energy_anomaly_detection/step1_etl.py`
from anywhere, or opening/running the file directly in an IDE).

Import the constants you need instead of hardcoding relative strings:
    from config import RAW_CSV, CLEAN_CSV, MODEL_DIR, PLOTS_TRAIN_DIR
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ── Data files ────────────────────────────────────────────────
DATASET_DIR  = str(BASE_DIR / "dataset")
RAW_DATA_CSV = str(BASE_DIR / "dataset" / "raw_data.csv")

ETL_OUTPUT_DIR = str(BASE_DIR / "etl_output")
CLEAN_CSV      = str(BASE_DIR / "etl_output" / "etl_clean.csv")
RAW_CSV        = str(BASE_DIR / "etl_output" / "etl_raw.csv")

# ── Model artifacts ───────────────────────────────────────────
MODEL_DIR        = str(BASE_DIR / "model")
BEST_MODEL_PT    = str(BASE_DIR / "model" / "best_model.pt")
SCALER_PKL       = str(BASE_DIR / "model" / "scaler.pkl")
THRESHOLD_NPY    = str(BASE_DIR / "model" / "threshold.npy")
HISTORY_NPZ      = str(BASE_DIR / "model" / "history.npz")  # np.savez: keys "train","val"
MODEL_INFO_TXT   = str(BASE_DIR / "model" / "model_info.txt")

# ── Results ───────────────────────────────────────────────────
RESULTS_DIR           = str(BASE_DIR / "results")
ANOMALY_RESULTS_CSV   = str(BASE_DIR / "results" / "anomaly_results.csv")
ANOMALY_EVENTS_CSV    = str(BASE_DIR / "results" / "anomaly_events.csv")
BASELINE_METRICS_CSV  = str(BASE_DIR / "results" / "baseline_metrics.csv")

# ── Plots ─────────────────────────────────────────────────────
PLOTS_DIR         = str(BASE_DIR / "plots")
PLOTS_TRAIN_DIR   = str(BASE_DIR / "plots" / "train")
PLOTS_DETECT_DIR  = str(BASE_DIR / "plots" / "detect")
PLOTS_TUNE_DIR    = str(BASE_DIR / "plots" / "tune")

# ── Latent sweep (self-contained experiment output) ──────────
LATENT_SWEEP_DIR       = str(BASE_DIR / "latent_sweep")
LATENT_SWEEP_PLOTS_DIR = str(BASE_DIR / "latent_sweep" / "plots")
LATENT_SWEEP_RESULTS_CSV = str(BASE_DIR / "latent_sweep" / "results.csv")


def ensure_dirs(*dirs):
    """Create directories if they don't exist yet (idempotent)."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
