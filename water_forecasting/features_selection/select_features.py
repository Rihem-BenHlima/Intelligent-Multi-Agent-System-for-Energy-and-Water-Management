"""
Feature Selection — Water Consumption
===================================================
Selects the most predictive features using a hybrid approach:
1. Mutual Information (Non-linear correlations)
2. XGBoost Feature Importance (Tree-based interactions)

Evaluates feature importance directly against the raw 'consumption_l' target
(no log1p transform — features and target are both on the raw L/h scale).
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.feature_selection import mutual_info_regression
from xgboost import XGBRegressor

# Number of features to keep
TOP_K = 15

def run():
    print("=" * 60)
    print(" FEATURE SELECTION (Mutual Info + XGBoost)")
    print("=" * 60)

    base_dir = Path(__file__).resolve().parent
    input_file = base_dir.parent / "features_engineering" / "output" / "features.csv"
    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        sys.exit(1)

    print(f"Loading: {input_file.name}...")
    df = pd.read_csv(input_file, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # 1. DEFINE TARGET & FEATURES TO EXCLUDE
    target_col = "consumption_l"

    if target_col not in df.columns:
        print(f"ERROR: '{target_col}' not found in the dataset.")
        print("Please rerun generate_features.py first.")
        sys.exit(1)

    # We drop metadata and the target to avoid Data Leakage
    cols_to_drop = ["datetime", target_col]

    # Optional: Drop ETL bookkeeping columns if they leaked through
    etl_cols = ["gap_tier", "is_imputed", "fill_method"]
    cols_to_drop.extend([c for c in etl_cols if c in df.columns])

    feature_cols = [c for c in df.columns if c not in cols_to_drop]

    # LEAKAGE FIX: score features using only the train split (same 70% cut
    # used later in train_lstm.py). Scoring on the full series would let
    # MI/XGBoost "see" val/test-period target values when deciding which
    # columns are predictive, biasing the selected feature set with
    # information the model isn't supposed to have yet.
    train_end = int(len(df) * 0.70)
    df_train = df.iloc[:train_end]

    # Handle missing values purely for the selection step (filling with
    # train-only medians, to match how train_gru.py fits its scaler)
    X = df_train[feature_cols].copy()
    X = X.fillna(X.median())
    y = df_train[target_col].values

    print(f"\nEvaluating {len(feature_cols)} features against '{target_col}' (train split only, n={len(df_train):,})...")

    # 2. MUTUAL INFORMATION
    print("  -> Computing Mutual Information...")
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns)
    # Normalize MI scores to 0-1 range for fair blending
    if mi_series.max() > 0:
        mi_series = mi_series / mi_series.max()

    # 3. XGBOOST IMPORTANCE
    print("  -> Computing XGBoost Importance...")
    xgb = XGBRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X, y)
    xgb_series = pd.Series(xgb.feature_importances_, index=X.columns)
    # XGB importances are already roughly 0-1, but normalize just in case
    if xgb_series.max() > 0:
        xgb_series = xgb_series / xgb_series.max()

    # 4. HYBRID SCORING (50% MI / 50% XGB)
    print("\n  -> Blending Scores...")
    hybrid_scores = (0.5 * mi_series) + (0.5 * xgb_series)
    hybrid_scores = hybrid_scores.sort_values(ascending=False)

    top_features = hybrid_scores.head(TOP_K).index.tolist()

    print(f"\nTop {TOP_K} Features Selected:")
    for i, (feat, score) in enumerate(hybrid_scores.head(TOP_K).items(), 1):
        print(f"  {i:2d}. {feat:<20} (Score: {score:.4f})")

    # 5. SAVE JSON SUMMARY
    summary = {
        "target_evaluated": target_col,
        "top_k": TOP_K,
        "selected_features": top_features,
        "scores": hybrid_scores.head(TOP_K).to_dict()
    }
    json_path = output_dir / "selected_features.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)

    # 6. SAVE FILTERED DATASET
    output_cols = ["datetime", target_col] + top_features
    # Ensure all columns actually exist (sanity check)
    output_cols = [c for c in output_cols if c in df.columns]

    df_selected = df[output_cols]
    
    csv_path = output_dir / "selected_features_data.csv"
    df_selected.to_csv(csv_path, index=False)

    print(f"\nOutputs generated successfully:")
    print(f"  - {json_path.name}")
    print(f"  - {csv_path.name}")


if __name__ == "__main__":
    run()