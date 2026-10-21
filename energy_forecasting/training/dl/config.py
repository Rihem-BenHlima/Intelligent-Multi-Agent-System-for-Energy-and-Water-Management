"""
config.py
=========
Shared configuration for deep learning model training pipelines (GRU, LSTM).
Defines paths, feature schemas, chronological validation splits, and evaluation thresholds.
"""

from pathlib import Path
import json

# Root path: points to energy_forecasting/
ROOT = Path(__file__).resolve().parent.parent.parent

# ── Data paths ────────────────────────────────────────────────────────────
# Primary: rows containing only the selected features
FEATURES_CSV = ROOT / "features_selection" / "output" / "features_final_data.csv"

# Fallback: full engineered feature set (used if feature selection not yet run)
if not FEATURES_CSV.exists():
    FEATURES_CSV = ROOT / "features_engineering" / "output" / "features.csv"

# Feature selection results — JSON with the final selected feature names
FEATURE_SELECTION_JSON = ROOT / "features_selection" / "output" / "feature_selection.json"

# Training directory root
TRAINING_DIR = ROOT / "training"

# ── Target & columns to never use as features ─────────────────────────────
TARGET    = "target_24h"
DROP_COLS = {"sn", "datetime", TARGET}

# ── Load selected feature names ───────────────────────────────────────────
def get_selected_features():
    """
    Load the final selected feature set from feature_selection.json.

    Returns
    -------
    list[str] | None
        List of feature column names, or None if the JSON file is absent.
    """
    if not FEATURE_SELECTION_JSON.exists():
        return None
    with open(FEATURE_SELECTION_JSON, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("final_features", None)

SELECTED_FEATURES = get_selected_features()

# ── Train / test split ────────────────────────────────────────────────────
TRAIN_RATIO = 0.80   # 80% of chronological data used for training
GAP_HOURS   = 24     # hours removed between train end and test start
                     # prevents look-ahead leakage across the boundary

# ── Walk-forward cross-validation folds ───────────────────────────────────
# Each tuple: (fold_name, train_end_date, test_start_date, test_end_date)
# A GAP_HOURS gap is applied on top of test_start inside each model.
WALK_FORWARD_FOLDS = [
    ("Fold1_Summer", "2025-06-30", "2025-07-01", "2025-08-31"),
    ("Fold2_Autumn", "2025-09-30", "2025-10-01", "2025-11-30"),
    ("Fold3_Winter", "2025-12-31", "2026-01-01", "2026-02-28"),
    ("Fold4_WinSpr", "2026-01-24", "2026-01-25", "2026-05-07"),
]

# ── Sample weights for chronological training ─────────────────────────────
# Linear ramp WEIGHT_MIN → WEIGHT_MAX so recent samples matter more.
WEIGHT_MIN = 0.2
WEIGHT_MAX = 1.0

# ── Evaluation thresholds (ASHRAE / BuildingsBench / KARN 2025) ──────────
THRESHOLDS = {
    "cv_rmse": {"excellent": 15.0, "good": 30.0},   # % — lower is better
    "r2":      {"excellent":  0.90, "good":  0.80},  # higher is better
    "smape":   {"excellent": 10.0, "good": 20.0},    # % — lower is better
}