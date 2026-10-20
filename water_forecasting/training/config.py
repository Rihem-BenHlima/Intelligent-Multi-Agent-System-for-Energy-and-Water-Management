"""
Shared configuration for all training scripts.
"""
from pathlib import Path

# ── Root anchor ──────────────────────────────────────────────────────────────
# This file lives in: test1/training/
# All paths below are derived from this anchor.
TRAINING_DIR = Path(__file__).resolve().parent
TEST1_DIR    = TRAINING_DIR.parent

# ── Input data ───────────────────────────────────────────────────────────────
FEATURES_CSV = TEST1_DIR / "features_engineering" / "output" / "features.csv"

# ── Feature selection output (from feature_selection/select_features.py) ─────
SELECTED_FEATURES_JSON = TEST1_DIR / "features_selection" / "output" / "selected_features.json"
