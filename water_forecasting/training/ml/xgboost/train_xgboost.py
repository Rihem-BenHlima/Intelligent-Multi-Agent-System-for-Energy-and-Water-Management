"""
XGBoost hurdle model — Office Water Consumption Forecasting
Stage 1: XGBClassifier (zero vs nonzero)
Stage 2: XGBRegressor split by working-hour vs off-hour regimes
Inputs : features_selection/output/selected_features_data.csv
Outputs: output/xgb_metrics.json, output/xgb_test_predictions.csv,
         feature importance CSVs, model/*.json, plots/
Split  : chronological 70/15/15.
Run    : python train_xgboost.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor, XGBClassifier
import shap

import argparse
import os

BASE_DIR = Path(__file__).resolve().parent
MAX_SEARCH_LEVELS = 5
OUTPUT_DIR_NAME = "output"


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def _find_file(filename, override):
    if override:
        p = Path(override).expanduser().resolve()
        if not p.exists():
            print(f"ERROR: explicitly specified path not found: {p}")
            sys.exit(1)
        return p

    direct = BASE_DIR / filename
    if direct.exists():
        return direct

    searched_roots = []
    d = BASE_DIR
    for _ in range(MAX_SEARCH_LEVELS):
        parent = d.parent
        if parent == d:
            break
        d = parent
        if d.parent == d:
            searched_roots.append(d)
            break
        searched_roots.append(d)
        matches = list(d.rglob(filename))
        matches = [m for m in matches if OUTPUT_DIR_NAME not in m.parts
                   or m.parent != BASE_DIR / OUTPUT_DIR_NAME]
        if matches:
            if len(matches) > 1:
                print(f"NOTE: multiple copies of {filename} found, using the first:")
                for m in matches:
                    print(f"    {m}")
            return matches[0]

    print(f"ERROR: could not locate {filename}.")
    print(f"  Searched: {BASE_DIR}")
    for r in searched_roots:
        print(f"  Searched (recursively): {r}")
    print(f"  Run select_features.py first, or pass an explicit path, e.g.:")
    print(f"    python {Path(__file__).name} --input path\\to\\selected_features_data.csv "
          f"--features-json path\\to\\selected_features.json")
    sys.exit(1)


_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--input", default=None)
_parser.add_argument("--features-json", default=None)
_cli_args, _ = _parser.parse_known_args()

_input_override = _cli_args.input or os.environ.get("XGB_INPUT_FILE")
_features_override = _cli_args.features_json or os.environ.get("XGB_FEATURES_JSON")

OUTPUT_DIR = BASE_DIR / OUTPUT_DIR_NAME
PLOT_DIR   = BASE_DIR / "plots"
MODEL_DIR  = BASE_DIR / "model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = _find_file("selected_features_data.csv", _input_override)
FEATURES_JSON = _find_file("selected_features.json", _features_override)

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15   # remaining 0.15 goes to test
RANDOM_STATE = 42
FORECAST_DAYS = 7   # number of days shown in the forecast plot

# Fixed display window for visual comparison (Monday-Sunday with typical office profile)
BEST_PLOT_WINDOW_START = pd.Timestamp("2017-11-27")
FLOOR_L_PER_H = 5.0  # snap small residual predictions to 0
MASE_SEASONAL_PERIOD = 24  # hours — "same hour, previous day" naive benchmark
FUTURE_FORECAST_HOURS = 24  # length of forward forecast


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def pbias(y_true, y_pred):
    """Percent Bias: 100 * sum(actual - pred) / sum(actual)."""
    denom = np.sum(y_true)
    if denom == 0:
        return np.nan
    return 100 * np.sum(y_true - y_pred) / denom


def seasonal_naive_mae(y_train, m=MASE_SEASONAL_PERIOD):
    """In-sample MAE of seasonal-naive forecast (y[t-m]) on training set."""
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) <= m:
        return np.nan
    return np.mean(np.abs(y_train[m:] - y_train[:-m]))


def mase(y_true, y_pred, naive_mae):
    """Mean Absolute Scaled Error (MASE < 1 beats seasonal-naive benchmark)."""
    if naive_mae == 0 or np.isnan(naive_mae):
        return np.nan
    return np.mean(np.abs(y_true - y_pred)) / naive_mae


def seasonal_naive_mse(y_train, m=MASE_SEASONAL_PERIOD):
    """In-sample MSE of seasonal-naive forecast (y[t-m]) on training set."""
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) <= m:
        return np.nan
    return np.mean((y_train[m:] - y_train[:-m]) ** 2)


def rmsse(y_true, y_pred, naive_mse):
    """Root Mean Squared Scaled Error (RMSSE < 1 beats seasonal-naive benchmark)."""
    if naive_mse == 0 or np.isnan(naive_mse):
        return np.nan
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse / naive_mse)


# ---------------------------------------------------------------------------
# Qualitative rating scales
# ---------------------------------------------------------------------------
def rate_pbias(v):
    """Moriasi et al. (2007) |PBIAS| ratings."""
    if np.isnan(v): return "n/a"
    av = abs(v)
    if av < 10:  return "Very good"
    if av < 15:  return "Good"
    if av < 25:  return "Satisfactory"
    return "Unsatisfactory"


def rate_mase(v):
    """Hyndman & Koehler (2006) MAE rating against naive benchmark."""
    if np.isnan(v): return "n/a"
    return ("Beats seasonal-naive benchmark" if v < 1.0
            else "Worse than seasonal-naive benchmark")


def rate_rmsse(v):
    """Hyndman & Koehler (2006) RMSSE rating against naive benchmark."""
    if np.isnan(v): return "n/a"
    return ("Beats seasonal-naive benchmark" if v < 1.0
            else "Worse than seasonal-naive benchmark")


def rate_r2(v):
    """R2 rating (Amasyali & El-Gohary 2018)."""
    if np.isnan(v): return "n/a"
    if v >= 0.90: return "Excellent"
    if v >= 0.75: return "Good"
    if v >= 0.50: return "Satisfactory"
    return "Unsatisfactory"


RATERS = {
    "PBIAS_pct": rate_pbias,
    "MASE":      rate_mase,
    "RMSSE":     rate_rmsse,
    "R2":        rate_r2,
}


def compute_metrics(y_true, y_pred, naive_mae, naive_mse):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "n":              len(y_true),
        "MAE_L_per_h":    mean_absolute_error(y_true, y_pred),
        "R2":             r2_score(y_true, y_pred),
        "PBIAS_pct":      pbias(y_true, y_pred),
        "MASE":           mase(y_true, y_pred, naive_mae),
        "RMSSE":          rmsse(y_true, y_pred, naive_mse),
    }


def print_metrics_table(all_metrics):
    splits = list(all_metrics.keys())
    keys = ["n", "MAE_L_per_h", "R2", "PBIAS_pct", "MASE", "RMSSE"]

    # Column widths: metric name | value | rating
    print(f"{'Metric':<14}  {'Value':>10}  {'Rating'}")
    print("-" * 52)
    for k in keys:
        for s in splits:
            v = all_metrics[s][k]
            val_str = f"{v:>10,.1f}" if k == "n" else f"{v:>10.4f}"
            rating  = f"  {RATERS[k](v)}" if k in RATERS else ""
            print(f"{k:<14}  {val_str}{rating}")


# ---------------------------------------------------------------------------
# Forward forecast
# ---------------------------------------------------------------------------
def forecast_next_horizon(df, target_col, feature_cols, clf, model_work, model_offhour,
                           horizon=FUTURE_FORECAST_HOURS):
    """Recursively forecast the `horizon` hours immediately after the last known timestamp in df."""
    dow_all  = df["datetime"].dt.dayofweek
    hour_all = df["datetime"].dt.hour
    door_lookup = df.groupby([dow_all, hour_all])["door_open_prob"].first()
    door_default = df["door_open_prob"].mean()

    dt_hist = list(df["datetime"])
    y_hist  = list(df[target_col].values.astype(float))
    last_dt = dt_hist[-1]

    out_rows = []
    for step in range(1, horizon + 1):
        t = last_dt + pd.Timedelta(hours=step)
        hour = t.hour
        dow  = t.dayofweek
        is_working_hour = int((dow < 5) and (8 <= hour < 18))

        row = {
            "hour":     hour,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "is_working_hour": is_working_hour,
            "lag_1h":   y_hist[-1],
            "lag_2h":   y_hist[-2],
            "lag_3h":   y_hist[-3],
            "lag_24h":  y_hist[-24],
            "lag_168h": y_hist[-168],
            "diff_1h":  y_hist[-1] - y_hist[-2],
            "diff_24h": y_hist[-1] - y_hist[-25],
            "rolling_mean_3h": np.mean(y_hist[-3:]),
            "rolling_mean_6h": np.mean(y_hist[-6:]),
            "ewma_6h": pd.Series(y_hist).ewm(span=6, adjust=False).mean().iloc[-1],
            "door_open_prob": door_lookup.get((dow, hour), door_default),
        }
        X_row = pd.DataFrame([row])[feature_cols]

        is_nonzero = clf.predict(X_row)[0]
        if is_nonzero == 0:
            pred = 0.0
        else:
            regressor = model_work if is_working_hour == 1 else model_offhour
            pred = float(np.clip(regressor.predict(X_row)[0], 0, None))
            if pred < FLOOR_L_PER_H:
                pred = 0.0

        out_rows.append({"datetime": t, "predicted_l": pred})
        dt_hist.append(t)
        y_hist.append(pred)

    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_forecast(test_df, pred_l, out_path, n_days=FORECAST_DAYS, future_df=None,
                   error_scale=None):
    """Plot actual vs predicted for a representative sample window plus future forecast."""
    INK      = "#0b0b0b"   # actual — near-black, the reference series
    ACCENT   = "#2a78d6"   # predicted — validated categorical slot 1 blue
    MUTED    = "#898781"   # axis/tick ink
    GRID     = "#e1e0d9"   # hairline gridline

    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "savefig.facecolor": "white",
        "font.family":       "sans-serif",
        "font.size":         10,
        "axes.edgecolor":    MUTED,
        "text.color":        INK,
        "axes.labelcolor":   INK,
        "xtick.color":       MUTED,
        "ytick.color":       MUTED,
    })

    window_start = BEST_PLOT_WINDOW_START
    if not (test_df["datetime"].min() <= window_start <= test_df["datetime"].max()):
        window_start = test_df["datetime"].iloc[int(len(test_df) * 0.25)].normalize()
    window_end   = window_start + pd.Timedelta(days=n_days)

    mask = (test_df["datetime"] >= window_start) & (test_df["datetime"] < window_end)
    win_df   = test_df[mask].copy()
    win_pred = pred_l[mask.values]

    has_future = future_df is not None and len(future_df) > 0
    plot_end = future_df["datetime"].iloc[-1] + pd.Timedelta(hours=1) if has_future else window_end

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(win_df["datetime"], win_df["consumption_l"],
            color=INK, linewidth=1.25, label="Actual", zorder=3,
            solid_capstyle="round")
    ax.plot(win_df["datetime"], win_pred,
            color=ACCENT, linewidth=1.25,
            label="XGBoost forecast", zorder=2,
            solid_capstyle="round")

    if has_future:
        # Bridge the gap so the future line connects to the last point of
        # the displayed window (not the dataset's true end, which may be
        # well past what's shown here)
        bridge_dt   = pd.concat([win_df["datetime"].iloc[-1:], future_df["datetime"]])
        bridge_pred = np.concatenate([[win_pred[-1]], future_df["predicted_l"].values])
        ax.plot(bridge_dt, bridge_pred,
                color=ACCENT, linewidth=1.25,
                zorder=2, solid_capstyle="round")

        # Forecast interval — half-width = test-set MAE * sqrt(step), so it
        # widens the further out the recursive forecast reaches (each step's
        # own error compounds on top of the steps feeding it), anchored at
        # zero width on the last known point. Deliberately not labeled as a
        # precise "95%" interval — MAE*sqrt(h) is a standard rough scale for
        # compounding step error, not a fitted quantile.
        mae = error_scale if error_scale else 0.0
        steps = np.arange(1, len(future_df) + 1)
        half_width = mae * np.sqrt(steps)
        band_dt    = bridge_dt
        band_upper = np.concatenate([[win_pred[-1]], future_df["predicted_l"].values + half_width])
        band_lower = np.concatenate([[win_pred[-1]], np.clip(future_df["predicted_l"].values - half_width, 0, None)])
        ax.fill_between(band_dt, band_lower, band_upper,
                         color=ACCENT, alpha=0.15, linewidth=0, zorder=1,
                         label="Forecast interval")

    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Consumption (L/h)", fontsize=10)
    ax.set_ylim(bottom=0)
    ax.set_xlim(window_start, plot_end)
    ax.grid(True, axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.grid(False, axis="x")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(axis="both", length=3, labelsize=9)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")

    ax.legend(
        loc="upper right", frameon=True, framealpha=0.9,
        edgecolor=GRID, facecolor="white",
        fontsize=9, handlelength=2.2,
    )

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")

    # Reset rcParams to avoid leaking into subsequent plots
    plt.rcParams.update({
        "axes.edgecolor":  plt.rcParamsDefault["axes.edgecolor"],
        "text.color":      plt.rcParamsDefault["text.color"],
        "axes.labelcolor": plt.rcParamsDefault["axes.labelcolor"],
        "xtick.color":     plt.rcParamsDefault["xtick.color"],
        "ytick.color":     plt.rcParamsDefault["ytick.color"],
    })


def compute_shap_values(model, X, sample_size=250):
    """Compute SHAP values for one regime regressor."""
    if len(X) == 0:
        return None, None
    print("  Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    X_sample = X.iloc[:min(sample_size, len(X))]
    return explainer.shap_values(X_sample), X_sample


def plot_shap(shap_values, X_sample, feature_cols, out_path, max_display=20):
    """SHAP beeswarm summary plot for combined regimes."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "savefig.facecolor":"white",
        "font.size": 10,
    })

    if shap_values is None or len(X_sample) == 0:
        print("  Skipping SHAP plot: no test rows")
        return

    fig, ax = plt.subplots(figsize=(10, 0.45 * min(max_display, len(feature_cols)) + 3))

    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_cols,
        max_display=max_display,
        show=False,
        plot_size=None,
        color_bar=True,
    )

    ax = plt.gca()
    ax.set_title(
        "XGBoost SHAP Summary Plot (t+1 estimator, test set, both regimes combined)",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    print("=" * 65)
    print("  XGBoost — Office Water Consumption Forecasting (Hurdle Model)")
    print("=" * 65)

    print(f"\nLoading data from     : {INPUT_FILE}")
    print(f"Loading features from : {FEATURES_JSON}")

    df = pd.read_csv(INPUT_FILE, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    with open(FEATURES_JSON) as f:
        feat_summary = json.load(f)
    feature_cols = feat_summary["selected_features"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    if "is_working_hour" not in feature_cols:
        print("ERROR: 'is_working_hour' is required as the regime-routing signal "
              "for the hurdle model but was not selected by select_features.py.")
        sys.exit(1)

    # Target: raw consumption_l — no log transform
    target_col = "consumption_l"

    n = len(df)
    train_end = int(n * TRAIN_FRAC)
    val_end   = int(n * (TRAIN_FRAC + VAL_FRAC))

    train_df = df.iloc[:train_end]
    val_df   = df.iloc[train_end:val_end]
    test_df  = df.iloc[val_end:]

    print(f"\nFeatures used ({len(feature_cols)}): {feature_cols}")
    print(f"Split sizes -> train: {len(train_df):,} | val: {len(val_df):,} | "
          f"test: {len(test_df):,}")
    print(f"Test  period: {test_df['datetime'].min()} -> {test_df['datetime'].max()}")

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val   = val_df[feature_cols]
    y_val   = val_df[target_col]
    X_test  = test_df[feature_cols]
    y_test  = test_df[target_col]

    naive_mae = seasonal_naive_mae(y_train, m=MASE_SEASONAL_PERIOD)
    naive_mse = seasonal_naive_mse(y_train, m=MASE_SEASONAL_PERIOD)
    print(f"\nMASE/RMSSE seasonal-naive scale (m={MASE_SEASONAL_PERIOD}h): "
          f"in-sample MAE={naive_mae:.4f} L/h, MSE={naive_mse:.4f} (L/h)^2")

    # ---- Stage 1: zero vs nonzero classifier ----
    print("\n[Stage 1] Training zero/nonzero classifier...")
    y_train_cls = (y_train > 0).astype(int)
    y_val_cls   = (y_val > 0).astype(int)

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric="logloss",
    )
    clf.fit(X_train, y_train_cls, eval_set=[(X_val, y_val_cls)], verbose=False)
    print(f"  Best iteration: {clf.best_iteration}")

    # ---- Stage 2: regime-routed regressors, nonzero rows only ----
    print("\n[Stage 2] Training regime-routed regressors on nonzero rows...")
    train_nonzero = train_df[train_df[target_col] > 0]
    val_nonzero   = val_df[val_df[target_col] > 0]

    def _fit_regime(name, mask_train, mask_val):
        Xtr = train_nonzero.loc[mask_train, feature_cols]
        ytr = train_nonzero.loc[mask_train, target_col]
        Xv  = val_nonzero.loc[mask_val, feature_cols]
        yv  = val_nonzero.loc[mask_val, target_col]
        m = XGBRegressor(
            n_estimators=1000,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            early_stopping_rounds=50,
            eval_metric="rmse",
        )
        m.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
        print(f"  [{name}] best iteration: {m.best_iteration}  "
              f"(train n={len(Xtr):,}, val n={len(Xv):,})")
        return m

    work_train_mask = train_nonzero["is_working_hour"] == 1
    work_val_mask   = val_nonzero["is_working_hour"] == 1
    model_work    = _fit_regime("working-hour regime", work_train_mask, work_val_mask)
    model_offhour = _fit_regime("off-hour regime",     ~work_train_mask, ~work_val_mask)

    # ---- Combine: classifier gate + regime-routed magnitude ----
    nonzero_pred_cls = clf.predict(X_test)
    test_work_mask   = (X_test["is_working_hour"] == 1).values
    pred_work    = np.clip(model_work.predict(X_test), 0, None)
    pred_offhour = np.clip(model_offhour.predict(X_test), 0, None)
    regime_pred  = np.where(test_work_mask, pred_work, pred_offhour)
    pred_l = np.where(nonzero_pred_cls == 0, 0.0, regime_pred)
    # Safety-net floor — redundant given the classifier gate but harmless
    pred_l = np.where(pred_l < FLOOR_L_PER_H, 0.0, pred_l)

    all_metrics = {}
    actual_l = y_test.values
    all_metrics["test"] = compute_metrics(actual_l, pred_l, naive_mae, naive_mse)

    cls_acc = (nonzero_pred_cls == (actual_l > 0).astype(int)).mean()
    print(f"\nStage-1 classifier accuracy (zero vs nonzero): {cls_acc*100:.2f}%")

    print("\n" + "=" * 65)
    print("  Evaluation (litres/hour)")
    print("=" * 65)
    print_metrics_table(all_metrics)

    print("\nInterpretation guide (Moriasi et al. 2007, Trans. ASABE 50(3):885-900):")
    print("  PBIAS < 10%  very good | 10-15%    good | 15-25%    satisfactory | >25%  unsatisfactory")
    print("\nMASE / RMSSE benchmark (Hyndman & Koehler, 2006):")
    print(f"  MASE / RMSSE < 1.0 beats the seasonal-naive (same-hour-yesterday, "
          f"m={MASE_SEASONAL_PERIOD}h) forecast")

    # ---- Save predictions CSV ----
    test_out = test_df[["datetime", "consumption_l"]].copy()
    test_out["predicted_l"]  = pred_l
    test_out["abs_error_l"]  = np.abs(test_out["consumption_l"] - test_out["predicted_l"])
    csv_path = OUTPUT_DIR / "xgb_test_predictions.csv"
    test_out.to_csv(csv_path, index=False)
    print(f"\nSaved -> {csv_path}")

    # ---- Save metrics JSON ----
    metrics_json = {}
    for split, split_metrics in all_metrics.items():
        clean = {mk: (None if isinstance(mv, float) and np.isnan(mv) else mv)
                 for mk, mv in split_metrics.items()}
        clean["ratings"] = {mk: RATERS[mk](split_metrics[mk]) for mk in RATERS}
        metrics_json[split] = clean
    metrics_json["stage1_classifier_accuracy_pct"] = float(cls_acc * 100)
    metrics_json["mase_naive_scale_L_per_h"] = float(naive_mae)
    metrics_json["rmsse_naive_scale_L_per_h2"] = float(naive_mse)
    json_path = OUTPUT_DIR / "xgb_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics_json, f, indent=4)
    print(f"Saved -> {json_path}")

    # ---- Save feature importance CSVs (one per stage) ----
    def _save_importance(name, model, path):
        importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        importance.to_csv(path, header=["importance"])
        print(f"Saved -> {path}")
        print(f"\nTop features — {name}:")
        for feat, val in importance.head(10).items():
            print(f"  {feat:<20} {val:.4f}")

    _save_importance("Stage 1 classifier (zero/nonzero)", clf,
                      OUTPUT_DIR / "xgb_classifier_feature_importance.csv")
    _save_importance("Stage 2 working-hour regressor", model_work,
                      OUTPUT_DIR / "xgb_regressor_work_feature_importance.csv")
    _save_importance("Stage 2 off-hour regressor", model_offhour,
                      OUTPUT_DIR / "xgb_regressor_offhour_feature_importance.csv")

    # ---- Save models ----
    clf.save_model(MODEL_DIR / "xgb_classifier.json")
    model_work.save_model(MODEL_DIR / "xgb_regressor_work.json")
    model_offhour.save_model(MODEL_DIR / "xgb_regressor_offhour.json")
    print(f"\nSaved -> {MODEL_DIR / 'xgb_classifier.json'}")
    print(f"Saved -> {MODEL_DIR / 'xgb_regressor_work.json'}")
    print(f"Saved -> {MODEL_DIR / 'xgb_regressor_offhour.json'}")

    # ---- Forward forecast beyond sample window ----
    _plot_window_start = BEST_PLOT_WINDOW_START
    if not (test_df["datetime"].min() <= _plot_window_start <= test_df["datetime"].max()):
        _plot_window_start = test_df["datetime"].iloc[int(len(test_df) * 0.25)].normalize()
    forecast_anchor = _plot_window_start + pd.Timedelta(days=FORECAST_DAYS)
    history_df = df[df["datetime"] < forecast_anchor].reset_index(drop=True)
    print(f"\nForecasting the next {FUTURE_FORECAST_HOURS}h beyond "
          f"{history_df['datetime'].iloc[-1]} (pretending nothing after that "
          f"point is known)...")
    future_df = forecast_next_horizon(
        history_df, target_col, feature_cols, clf, model_work, model_offhour,
        horizon=FUTURE_FORECAST_HOURS,
    )
    print(future_df.to_string(index=False))

    # ---- Forecast plot (combined hurdle prediction + forward forecast) ----
    print("\nGenerating forecast plot...")
    error_scale = float(np.mean(np.abs(actual_l - pred_l)))  # test-set MAE
    plot_forecast(
        test_df.reset_index(drop=True),
        pred_l,
        PLOT_DIR / "xgb_forecast_plot.png",
        future_df=future_df,
        error_scale=error_scale,
    )

    # ---- SHAP summary plot (both regime regressors combined) ----
    print("Generating SHAP summary plot...")
    shap_work, X_shap_work = compute_shap_values(
        model_work, X_test[test_work_mask].reset_index(drop=True))
    shap_offhour, X_shap_offhour = compute_shap_values(
        model_offhour, X_test[~test_work_mask].reset_index(drop=True))

    shap_parts = [s for s in (shap_work, shap_offhour) if s is not None]
    X_parts    = [x for x in (X_shap_work, X_shap_offhour) if x is not None]
    combined_shap = np.concatenate(shap_parts, axis=0) if shap_parts else None
    combined_X    = pd.concat(X_parts, ignore_index=True) if X_parts else pd.DataFrame()

    plot_shap(
        combined_shap,
        combined_X,
        feature_cols,
        PLOT_DIR / "xgb_shap_summary.png",
    )

    print("\nDone.")
    return (clf, model_work, model_offhour), all_metrics


if __name__ == "__main__":
    run()