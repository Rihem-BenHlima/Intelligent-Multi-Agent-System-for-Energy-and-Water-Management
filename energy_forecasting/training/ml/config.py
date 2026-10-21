"""
config.py
=========
Shared configuration for Machine Learning model training pipelines
(CatBoost, LightGBM, Random Forest, XGBoost).
Defines paths, feature schemas, cross-validation folds, and evaluation thresholds.
"""

from pathlib import Path
import json
import pandas as pd
from typing import List, Dict, Any, Optional

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent

FEATURES_CSV = ROOT / "features_selection" / "output" / "features_final_data.csv"

if not FEATURES_CSV.exists():
    FEATURES_CSV = ROOT / "features_engineering" / "output" / "features.csv"

FEATURE_SELECTION_JSON = ROOT / "features_selection" / "output" / "feature_selection.json"

TRAINING_DIR = ROOT / "training"

# ── Target & index columns ─────────────────────────────────────────────────
TARGET    = "target_24h"
DROP_COLS = {"sn", "datetime", TARGET}

# ── Load selected feature names ────────────────────────────────────────────
def get_selected_features():
    if not FEATURE_SELECTION_JSON.exists():
        return None
    with open(FEATURE_SELECTION_JSON) as f:
        data = json.load(f)
    return data.get("final_features", None)

SELECTED_FEATURES = get_selected_features()

# ── Evaluation strategy ────────────────────────────────────────────────────
TRAIN_RATIO = 0.80
GAP_HOURS   = 24

# Walk-forward cross-validation folds (seasonal, non-overlapping)
WALK_FORWARD_FOLDS = [
    ("Fold1_Summer", "2025-06-30", "2025-07-01", "2025-08-31"),
    ("Fold2_Autumn", "2025-09-30", "2025-10-01", "2025-11-30"),
    ("Fold3_Winter", "2025-12-31", "2026-01-01", "2026-02-28"),
    ("Fold4_Spring", "2026-02-28", "2026-03-01", "2026-05-07"),
    ("Fold5_Summer2026", "2026-05-31", "2026-06-01", "2026-08-31"),
]

# ── Sample weighting ────────────────────────────────────────────────────────
# Recency weighting ramp
WEIGHT_MIN = 0.2
WEIGHT_MAX = 1.0

# Summer season multiplier (Jun-Sep) for models supporting seasonal upweighting
SUMMER_WEIGHT  = 8.0
SUMMER_MONTHS  = {6, 7, 8, 9}

# ── Evaluation thresholds (ASHRAE G14 / Literature benchmarks) ─────────────
THRESHOLDS = {
    "cv_rmse": {"excellent": 20.0, "good": 30.0},  # % (IPMVP / ASHRAE G14 hourly)
    "r2":      {"excellent": 0.90, "good": 0.75},  # Building energy literature standard
    "smape":   {"excellent": 10.0, "good": 20.0},  # % (Lewis scale)
}

# Visualization window parameters for forecast plots
FORECAST_DAY  = "2026-08-31"
HISTORY_HOURS = 168


# ── Walk-Forward Fold Builder Function ──────────────────────────────────────
def build_walk_forward_folds(
    df: pd.DataFrame,
    train_months: int = 6,
    test_months: int = 1,
    step_months: int = 1,
    min_train_rows: int = 2000,
    gap_hours: int = None,
) -> List[Dict[str, Any]]:
    """
    Build dynamic walk-forward folds for time series cross-validation.
    
    Automatically generates folds from the available date range.
    Also includes the hardcoded folds from WALK_FORWARD_FOLDS for backward compatibility.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'datetime' column
    train_months : int
        Number of months to use for training (default 6)
    test_months : int
        Number of months to use for testing (default 1)
    step_months : int
        Number of months to slide the window (default 1)
    min_train_rows : int
        Minimum rows required to keep a fold (default 2000)
    gap_hours : int
        Gap between train and test (defaults to GAP_HOURS)
    
    Returns
    -------
    List of fold definitions, each with:
        - fold: str (name/identifier)
        - train_end: datetime or str
        - test_start: datetime or str
        - test_end: datetime or str
        - cold_start: bool (True if test season not seen in training)
    """
    if gap_hours is None:
        gap_hours = GAP_HOURS
    
    df = df.sort_values("datetime").reset_index(drop=True)
    
    # First, include the hardcoded folds
    folds = []
    
    for fold_name, train_end_str, test_start_str, test_end_str in WALK_FORWARD_FOLDS:
        # Determine if this is a cold start fold
        # Cold start means the test season wasn't well-represented in training
        # For summer test folds, check if previous summer was in training
        is_cold_start = False
        
        if "Summer" in fold_name:
            # Extract year from test start
            try:
                test_year = int(test_start_str[:4])
                # Cold start if this is the first summer in the dataset
                # or if training didn't include previous summer
                first_date = df["datetime"].min()
                if test_year <= first_date.year:
                    is_cold_start = True
            except:
                pass
        
        folds.append({
            "fold": fold_name,
            "train_end": train_end_str,
            "test_start": test_start_str,
            "test_end": test_end_str,
            "cold_start": is_cold_start,
        })
    
    # Optional: Also generate dynamic folds from the data
    # This creates additional folds if the hardcoded ones don't cover the full range
    
    # Find all unique year-month periods
    df["year_month"] = df["datetime"].dt.to_period("M")
    unique_months = sorted(df["year_month"].unique())
    
    if len(unique_months) >= train_months + test_months:
        # Generate dynamic folds, but skip duplicates with hardcoded ones
        hardcoded_test_ranges = set()
        for f in folds:
            hardcoded_test_ranges.add((f["test_start"], f["test_end"]))
        
        for i in range(len(unique_months) - (train_months + test_months) + 1):
            # Define training months
            train_months_list = unique_months[i : i + train_months]
            train_end = train_months_list[-1].end_time.strftime("%Y-%m-%d")
            
            # Define test months (with gap)
            test_months_list = unique_months[i + train_months + 1 : i + train_months + 1 + test_months]
            if len(test_months_list) < test_months:
                continue
            
            test_start = test_months_list[0].start_time.strftime("%Y-%m-%d")
            test_end = test_months_list[-1].end_time.strftime("%Y-%m-%d")
            
            # Skip if this test range already exists in hardcoded folds
            if (test_start, test_end) in hardcoded_test_ranges:
                continue
            
            # Create fold name
            season_name = _get_season_name(test_months_list[0].month)
            fold_name = f"Fold_Dynamic_{season_name}_{i+1}"
            
            # Check if training includes same season
            train_seasons = {m.month for m in train_months_list}
            test_season = test_months_list[0].month
            is_cold_start = test_season not in train_seasons
            
            folds.append({
                "fold": fold_name,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "cold_start": is_cold_start,
            })
    
    # Remove the temporary year_month column
    df.drop(columns=["year_month"], inplace=True)
    
    return folds


def _get_season_name(month: int) -> str:
    """Helper function to get season name from month"""
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"


# Optional: Also add a function to convert string dates to datetime objects
def parse_fold_dates(fold: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert string dates in a fold definition to datetime objects.
    
    Parameters
    ----------
    fold : dict
        Fold definition with 'train_end', 'test_start', 'test_end' as strings
    
    Returns
    -------
    Fold definition with datetime objects
    """
    result = fold.copy()
    result["train_end"] = pd.to_datetime(result["train_end"])
    result["test_start"] = pd.to_datetime(result["test_start"])
    result["test_end"] = pd.to_datetime(result["test_end"])
    return result


# For backward compatibility with scripts that expect WALK_FORWARD_FOLDS as list of tuples
# But also support the function
def get_walk_forward_folds(df: pd.DataFrame = None) -> List:
    """
    Returns walk-forward folds. If DataFrame is provided, generates dynamic folds.
    Otherwise returns the hardcoded folds as tuples (for backward compatibility).
    """
    if df is not None:
        folds = build_walk_forward_folds(df)
        # Convert to the expected format if needed
        return [(f["fold"], f["train_end"], f["test_start"], f["test_end"]) for f in folds]
    return WALK_FORWARD_FOLDS