r"""
config.py — central path configuration for the water anomaly pipeline.
All paths derived from BASE_DIR (__file__). Edit WATER_CSV here; step1
auto-converts consumption_m3→consumption_l (×1000) if needed.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Input
DATASET_DIR = str(BASE_DIR / "dataset")
WATER_CSV   = str(BASE_DIR / "dataset" / "water_consumption_office.csv")

# ETL outputs
ETL_OUTPUT_DIR = str(BASE_DIR / "etl_output")
CLEAN_CSV      = str(BASE_DIR / "etl_output" / "etl_clean.csv")
RAW_CSV        = str(BASE_DIR / "etl_output" / "etl_raw.csv")

# Model
MODEL_DIR = str(BASE_DIR / "model")

# Plots
TRAIN_PLOT_DIR  = str(BASE_DIR / "plots" / "train")
TUNE_PLOT_DIR   = str(BASE_DIR / "plots" / "tune")
DETECT_PLOT_DIR = str(BASE_DIR / "plots" / "detect")

# Results
RESULT_DIR = str(BASE_DIR / "results")

# Latent sweep
LATENT_SWEEP_DIR       = str(BASE_DIR / "latent_sweep_water")
LATENT_SWEEP_PLOTS_DIR = str(BASE_DIR / "latent_sweep_water" / "plots")
