"""
LightGBM model — Office Water Consumption Forecasting
Trains on Top-K features selected by select_features.py.
Inputs : features_selection/output/selected_features_data.csv
Outputs: output/lgb_metrics.json, output/lgb_test_predictions.csv,
         output/lgb_feature_importance.csv, model/lgb_model.txt,
         plots/lgb_forecast_plot.png, plots/lgb_shap_summary.png
Split  : chronological 70/15/15.
Run    : python train_lightgbm.py
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
import lightgbm as lgb
import shap

import argparse
import os

BASE_DIR        = Path(__file__).resolve().parent
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

_input_override    = _cli_args.input or os.environ.get("LGB_INPUT_FILE")
_features_override = _cli_args.features_json or os.environ.get("LGB_FEATURES_JSON")

OUTPUT_DIR = BASE_DIR / OUTPUT_DIR_NAME
PLOT_DIR   = BASE_DIR / "plots"
MODEL_DIR  = BASE_DIR / "model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE    = _find_file("selected_features_data.csv", _input_override)
FEATURES_JSON = _find_file("selected_features.json", _features_override)

TRAIN_FRAC    = 0.70
VAL_FRAC      = 0.15
RANDOM_STATE  = 42
FORECAST_DAYS = 7

# Fixed display window for visual comparison (Monday-Sunday with typical office profile)
BEST_PLOT_WINDOW_START = pd.Timestamp("2017-11-27")
FLOOR_L_PER_H = 5.0  # snap small residual predictions to 0
MASE_SEASONAL_PERIOD = 24  # hours — "same hour, previous day" naive benchmark
FUTURE_FORECAST_HOURS = 24  # length of forward forecast


# ---------------------------------------------------------------------------
# Forward forecast
# ---------------------------------------------------------------------------
def forecast_next_horizon(df, target_col, feature_cols, model, horizon=FUTURE_FORECAST_HOURS):
    """Recursively forecast the `horizon` hours immediately after the last known timestamp in df."""
    dow_all  = df["datetime"].dt.dayofweek
    hour_all = df["datetime"].dt.hour
    door_lookup  = df.groupby([dow_all, hour_all])["door_open_prob"].first()
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

        pred = float(np.clip(model.predict(X_row)[0], 0, None))
        if pred < FLOOR_L_PER_H:
            pred = 0.0

        out_rows.append({"datetime": t, "predicted_l": pred})
        dt_hist.append(t)
        y_hist.append(pred)

    return pd.DataFrame(out_rows)


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
# Rating scales
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


RATERS = {"PBIAS_pct": rate_pbias, "MASE": rate_mase, "RMSSE": rate_rmsse, "R2": rate_r2}


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
    keys = ["n", "MAE_L_per_h", "R2", "PBIAS_pct", "MASE", "RMSSE"]
    print(f"{'Metric':<14}  {'Value':>10}  {'Rating'}")
    print("-" * 52)
    for k in keys:
        for s in all_metrics:
            v       = all_metrics[s][k]
            val_str = f"{v:>10,.1f}" if k == "n" else f"{v:>10.4f}"
            rating  = f"  {RATERS[k](v)}" if k in RATERS else ""
            print(f"{k:<14}  {val_str}{rating}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_forecast(test_df, pred_l, out_path, n_days=FORECAST_DAYS, future_df=None,
                   error_scale=None):
    """Plot actual vs predicted for a representative sample window plus future forecast."""
    INK    = "#0b0b0b"   # actual — near-black, the reference series
    ACCENT = "#2a78d6"   # predicted — validated categorical slot 1 blue
    MUTED  = "#898781"   # axis/tick ink
    GRID   = "#e1e0d9"   # hairline gridline

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

    mask     = (test_df["datetime"] >= window_start) & (test_df["datetime"] < window_end)
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
            label="LightGBM forecast", zorder=2,
            solid_capstyle="round")

    if has_future:
        bridge_dt   = pd.concat([win_df["datetime"].iloc[-1:], future_df["datetime"]])
        bridge_pred = np.concatenate([[win_pred[-1]], future_df["predicted_l"].values])
        ax.plot(bridge_dt, bridge_pred,
                color=ACCENT, linewidth=1.25,
                zorder=2, solid_capstyle="round")

        mae = error_scale if error_scale else 0.0
        steps = np.arange(1, len(future_df) + 1)
        half_width = mae * np.sqrt(steps)
        band_upper = np.concatenate([[win_pred[-1]], future_df["predicted_l"].values + half_width])
        band_lower = np.concatenate([[win_pred[-1]], np.clip(future_df["predicted_l"].values - half_width, 0, None)])
        ax.fill_between(bridge_dt, band_lower, band_upper,
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


def plot_shap(model, X_test, feature_cols, out_path, max_display=20):
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "savefig.facecolor":"white",
        "font.size": 10,
    })

    print("  Computing SHAP values (TreeExplainer)...")
    explainer   = shap.TreeExplainer(model)
    sample_size = min(500, len(X_test))
    X_sample    = X_test.iloc[:sample_size]
    shap_values = explainer.shap_values(X_sample)

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
        "LightGBM SHAP Summary Plot (t+1 estimator, test set)",
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
    print("  LightGBM — Office Water Consumption Forecasting")
    print("=" * 65)

    print(f"\nLoading data from     : {INPUT_FILE}")
    print(f"Loading features from : {FEATURES_JSON}")

    df = pd.read_csv(INPUT_FILE, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    with open(FEATURES_JSON) as f:
        feat_summary = json.load(f)
    feature_cols = feat_summary["selected_features"]
    feature_cols = [c for c in feature_cols if c in df.columns]

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

    # LightGBM Dataset format
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval   = lgb.Dataset(X_val,   label=y_val, reference=dtrain)

    params = {
        "objective":        "regression",
        "metric":           "rmse",
        "learning_rate":    0.03,
        "num_leaves":       31,
        "max_depth":        -1,         # unlimited — controlled by num_leaves
        "min_child_samples":20,         # minimum data in a leaf
        "feature_fraction": 0.8,        # subsample features per tree
        "bagging_fraction": 0.8,        # subsample rows per tree
        "bagging_freq":     1,
        "lambda_l2":        1.0,
        "verbose":          -1,
        "seed":             RANDOM_STATE,
        "n_jobs":           -1,
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=-1),  # suppress per-iteration output
    ]

    print("\nTraining LightGBM (raw L/h target, early stopping on val RMSE)...")
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=callbacks,
    )
    print(f"Best iteration: {model.best_iteration}")

    naive_mae = seasonal_naive_mae(y_train, m=MASE_SEASONAL_PERIOD)
    naive_mse = seasonal_naive_mse(y_train, m=MASE_SEASONAL_PERIOD)
    print(f"\nMASE/RMSSE seasonal-naive scale (m={MASE_SEASONAL_PERIOD}h): "
          f"in-sample MAE={naive_mae:.4f} L/h, MSE={naive_mse:.4f} (L/h)^2")

    pred_l   = np.clip(model.predict(X_test), 0, None)
    pred_l   = np.where(pred_l < FLOOR_L_PER_H, 0.0, pred_l)
    actual_l = test_df[target_col].values

    all_metrics = {"test": compute_metrics(actual_l, pred_l, naive_mae, naive_mse)}

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
    test_out["predicted_l"] = pred_l
    test_out["abs_error_l"] = np.abs(actual_l - pred_l)
    csv_path = OUTPUT_DIR / "lgb_test_predictions.csv"
    test_out.to_csv(csv_path, index=False)
    print(f"\nSaved -> {csv_path}")

    # ---- Save metrics JSON ----
    metrics_json = {}
    for split, split_metrics in all_metrics.items():
        clean = {mk: (None if isinstance(mv, float) and np.isnan(mv) else mv)
                 for mk, mv in split_metrics.items()}
        clean["ratings"] = {mk: RATERS[mk](split_metrics[mk]) for mk in RATERS}
        metrics_json[split] = clean
    metrics_json["mase_naive_scale_L_per_h"] = float(naive_mae)
    metrics_json["rmsse_naive_scale_L_per_h2"] = float(naive_mse)
    json_path = OUTPUT_DIR / "lgb_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics_json, f, indent=4)
    print(f"Saved -> {json_path}")

    # ---- Save feature importance CSV ----
    importance = pd.Series(
        model.feature_importance(importance_type="gain"),
        index=feature_cols
    ).sort_values(ascending=False)
    imp_path = OUTPUT_DIR / "lgb_feature_importance.csv"
    importance.to_csv(imp_path, header=["importance"])
    print(f"Saved -> {imp_path}")
    print("\nTop features by LightGBM gain importance:")
    total = importance.sum()
    for feat, val in importance.items():
        print(f"  {feat:<20} {val/total:.4f}")

    # ---- Save model ----
    model_path = MODEL_DIR / "lgb_model.txt"
    model.save_model(str(model_path))
    print(f"Saved -> {model_path}")

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
        history_df, target_col, feature_cols, model,
        horizon=FUTURE_FORECAST_HOURS,
    )
    print(future_df.to_string(index=False))

    # ---- Forecast plot ----
    print("\nGenerating forecast plot...")
    error_scale = float(np.mean(np.abs(actual_l - pred_l)))  # test-set MAE
    plot_forecast(
        test_df.reset_index(drop=True),
        pred_l,
        PLOT_DIR / "lgb_forecast_plot.png",
        future_df=future_df,
        error_scale=error_scale,
    )

    # ---- SHAP summary plot ----
    print("Generating SHAP summary plot...")
    plot_shap(
        model,
        X_test.reset_index(drop=True),
        feature_cols,
        PLOT_DIR / "lgb_shap_summary.png",
    )

    print("\nDone.")
    return model, all_metrics


if __name__ == "__main__":
    run()