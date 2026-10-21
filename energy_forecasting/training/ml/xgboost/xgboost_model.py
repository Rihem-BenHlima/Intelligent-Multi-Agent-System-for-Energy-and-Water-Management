"""
xgboost_model.py
================
XGBoost — 24h-ahead Building Energy Forecast
Single smart meter, Tunisia (hourly, Oct 2024 – Aug 2026)

Evaluation: 5-fold walk-forward CV (seasonal, non-overlapping).
Metrics: CV-RMSE, R², sMAPE, MAE, RMSE.
Outputs: xgboost_production.joblib, evaluation_summary.json, forecast_7day.png, shap_summary.png.
"""

import sys; sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import json
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from training.ml.config import (
    FEATURES_CSV, TARGET, DROP_COLS,
    TRAIN_RATIO, GAP_HOURS,
    WALK_FORWARD_FOLDS,
    WEIGHT_MIN, WEIGHT_MAX,
    SUMMER_MONTHS,
    THRESHOLDS,
    SELECTED_FEATURES,
    FEATURE_SELECTION_JSON,
    FORECAST_DAY,
    HISTORY_HOURS,
)

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR = Path(__file__).resolve().parent / "plot"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

PARAMS = dict(
    n_estimators      = 3000,
    learning_rate     = 0.01,
    max_depth         = 8,
    min_child_weight  = 3,
    subsample         = 0.85,
    colsample_bytree  = 0.80,
    colsample_bylevel = 0.80,
    reg_alpha         = 0.1,
    reg_lambda        = 2.0,
    gamma             = 0.1,
    random_state      = 42,
    n_jobs            = -1,
    tree_method       = "hist",
    verbosity         = 0,
)

MODEL_COLOR = "#1565C0"


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def make_weights(n: int) -> np.ndarray:
    """Compute plain recency weights (linear ramp)."""
    return np.linspace(WEIGHT_MIN, WEIGHT_MAX, n)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_pred  = np.clip(y_pred, 0, None)
    r2      = float(r2_score(y_true, y_pred))
    rmse    = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae     = float(mean_absolute_error(y_true, y_pred))
    smape   = float(np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    ) * 100)
    cv_rmse = rmse / float(np.mean(y_true)) * 100
    return dict(r2=r2, rmse=rmse, mae=mae, smape=smape, cv_rmse=cv_rmse)


def _tag(metric: str, value: float) -> str:
    th = THRESHOLDS.get(metric)
    if th is None:
        return ""
    if metric == "r2":
        if value >= th["excellent"]: return "✅ excellent"
        if value >= th["good"]:      return "✅ good"
        return "⚠️  below 0.80"
    else:
        if value <= th["excellent"]: return "✅ excellent"
        if value <= th["good"]:      return "✅ good"
        return "⚠️  above threshold"


def print_metrics(m: dict, label: str = ""):
    tag = f"  [{label}]" if label else ""
    print(f"\n{'='*62}")
    print(f"  RESULTS{tag}")
    print(f"{'='*62}")
    print(f"  CV-RMSE  = {m['cv_rmse']:6.2f}%    {_tag('cv_rmse', m['cv_rmse'])}")
    print(f"  R²       = {m['r2']:6.4f}     {_tag('r2', m['r2'])}")
    print(f"  sMAPE    = {m['smape']:6.2f}%    {_tag('smape', m['smape'])}")
    print(f"  MAE      = {m['mae']:6.4f} kWh")
    print(f"  RMSE     = {m['rmse']:6.4f} kWh")
    print(f"{'='*62}")


def monthly_breakdown(dates, y_true: np.ndarray, y_pred: np.ndarray):
    df = pd.DataFrame({
        "datetime": pd.to_datetime(dates),
        "actual":   y_true,
        "pred":     np.clip(y_pred, 0, None),
    })
    df["ym"] = df["datetime"].dt.to_period("M").astype(str)
    print(f"\n  Monthly breakdown:")
    print(f"  {'Month':<10} {'CV-RMSE':>9} {'R²':>8} {'sMAPE':>8} {'MAE':>9}  {'N':>5}")
    print("  " + "─" * 56)
    for ym, g in df.groupby("ym"):
        if len(g) < 20:
            continue
        m    = compute_metrics(g["actual"].values, g["pred"].values)
        flag = " ✓" if m["r2"] >= 0.70 else " ✗"
        print(f"  {ym:<10} {m['cv_rmse']:>8.2f}% {m['r2']:>8.4f}"
              f" {m['smape']:>7.2f}% {m['mae']:>8.4f}  {len(g):>5}{flag}")


# ═══════════════════════════════════════════════════════════════════════════
# FORECAST PLOT
# ═══════════════════════════════════════════════════════════════════════════

def plot_forecast(dates_te, y_te: np.ndarray, y_pred: np.ndarray,
                  label: str, out_path: Path, zoom_hours: int = 168):
    """Actual vs predicted chart, styled to match the water_forecasting
    XGBoost template: a plain white ground, black actual / blue model line,
    no title, clean spines, hairline grid, and a boxed legend placed clear
    of the data — Hyndman's `forecast` R package convention (black=actual,
    blue=forecast — pkg.robjhyndman.com/forecast)."""
    INK    = "#0b0b0b"   # actual — near-black, the reference series
    ACCENT = "#2a78d6"   # predicted/forecast — validated categorical slot 1 blue
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

    # Shift x-axis by +24h: rows carry prediction time t, target is t+24h.
    dates_te = pd.to_datetime(dates_te) + pd.Timedelta(hours=24)
    y_pred   = np.clip(y_pred, 0, None)
    n        = len(y_te)

    hist_end   = n - 24
    hist_start = max(0, hist_end - zoom_hours)

    d_hist = dates_te[hist_start : hist_end + 1]
    a_hist = y_te[hist_start : hist_end + 1]
    p_hist = y_pred[hist_start : hist_end + 1]
    d_fore = dates_te[hist_end : hist_end + 25]
    p_fore = y_pred[hist_end : hist_end + 25]

    resid_std = np.std(a_hist - p_hist)
    ci_up  = p_fore + 1.645 * resid_std
    ci_lo  = np.clip(p_fore - 1.645 * resid_std, 0, None)

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.fill_between(d_fore, ci_lo, ci_up,
                    color=ACCENT, alpha=0.15, linewidth=0, zorder=1,
                    label="forecast interval")
    ax.plot(d_hist, a_hist, color=INK, linewidth=1.25, zorder=3,
            solid_capstyle="round", label="Actual")
    ax.plot(d_hist, p_hist, color=ACCENT, linewidth=1.25, zorder=2,
            solid_capstyle="round", label="XGBoost forecast")
    ax.plot(d_fore, p_fore, color=ACCENT, linewidth=1.25, zorder=2,
            solid_capstyle="round")

    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Energy Consumption (kWh)", fontsize=10)
    ax.set_ylim(bottom=0)
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
        loc="lower left", bbox_to_anchor=(0.0, 1.02), frameon=True, framealpha=0.9,
        edgecolor=GRID, facecolor="white",
        fontsize=9, handlelength=2.2,
    )
    fig.subplots_adjust(top=0.85)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Plot → {out_path.name}")

    # Reset matplotlib rcParams to defaults.
    plt.rcParams.update({
        "axes.edgecolor":  plt.rcParamsDefault["axes.edgecolor"],
        "text.color":      plt.rcParamsDefault["text.color"],
        "axes.labelcolor": plt.rcParamsDefault["axes.labelcolor"],
        "xtick.color":     plt.rcParamsDefault["xtick.color"],
        "ytick.color":     plt.rcParamsDefault["ytick.color"],
    })


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 62)
    print("  XGBoost — 24h Building Energy Forecast")
    print("=" * 62)

    df = pd.read_csv(FEATURES_CSV)

    # ── Auto-detect datetime column ────────────────────────────────────────
    dt_col = None
    for col in df.columns:
        if col.lower() in ("datetime", "date", "timestamp", "time", "dt", "datetime_local"):
            dt_col = col
            break
    if dt_col is None:
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    pd.to_datetime(df[col].iloc[:5], errors="raise")
                    dt_col = col
                    break
                except Exception:
                    continue
    if dt_col is None:
        raise ValueError(f"No datetime column found. Columns: {list(df.columns)}")

    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values(dt_col).reset_index(drop=True)
    if dt_col != "datetime":
        df = df.rename(columns={dt_col: "datetime"})

    if SELECTED_FEATURES is None:
        raise FileNotFoundError(
            "feature_selection.json not found or missing 'final_features' key.\n"
            f"  Expected at: {FEATURE_SELECTION_JSON}"
        )
    feat_cols = [c for c in SELECTED_FEATURES if c in df.columns]
    missing = [c for c in SELECTED_FEATURES if c not in df.columns]
    if missing:
        print(f"  WARNING: {len(missing)} selected features not found in CSV: {missing}")
    if len(feat_cols) == 0:
        raise ValueError("None of the selected features are present in the data CSV.")
    print(f"  Using {len(feat_cols)} selected features (from feature_selection.json)")

    print(f"\n  Rows      : {len(df):,}")
    print(f"  Features  : {len(feat_cols)}")
    print(f"  Date range: {df['datetime'].min().date()} → {df['datetime'].max().date()}")
    print(f"  Target    : mean={df[TARGET].mean():.4f}  std={df[TARGET].std():.4f} kWh")

    # ── Summer stats in training data ──────────────────────────────────────
    summer_mask_all = df["datetime"].dt.month.isin(SUMMER_MONTHS)
    print(f"  Summer rows (all data): {summer_mask_all.sum():,} "
          f"({100*summer_mask_all.mean():.1f}%)")

    # ══════════════════════════════════════════════════════════════════════
    # SECONDARY — 5-fold walk-forward CV
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*62}")
    print("  SECONDARY — 5-fold Walk-Forward CV")
    print(f"{'─'*62}")

    fold_results = []
    pooled_dates, pooled_true, pooled_pred = [], [], []

    for fname, train_end, test_start, test_end in WALK_FORWARD_FOLDS:
        gap_start = pd.Timestamp(test_start) + pd.Timedelta(hours=GAP_HOURS)
        tr_mask   = df["datetime"] <= train_end
        te_mask   = (df["datetime"] >= gap_start) & (df["datetime"] <= test_end)

        if tr_mask.sum() < 200 or te_mask.sum() < 50:
            print(f"\n  {fname}: skipped (insufficient data)")
            continue

        Xf_tr = df[feat_cols][tr_mask].values
        yf_tr = df[TARGET][tr_mask].values
        Xf_te = df[feat_cols][te_mask].values
        yf_te = df[TARGET][te_mask].values
        df_dates = df["datetime"][te_mask].values

        wf = make_weights(tr_mask.sum())
        mf = XGBRegressor(**PARAMS)
        mf.fit(Xf_tr, yf_tr, sample_weight=wf)
        pf   = mf.predict(Xf_te)
        mets = compute_metrics(yf_te, pf)
        fold_results.append({"fold": fname, **mets})

        pooled_dates.append(df_dates)
        pooled_true.append(yf_te)
        pooled_pred.append(pf)

        print(f"\n  {fname}")
        print(f"    Train : {df['datetime'][tr_mask].min().date()} → "
              f"{df['datetime'][tr_mask].max().date()} ({tr_mask.sum():,})")
        print(f"    Test  : {pd.Timestamp(gap_start).date()} → "
              f"{test_end} ({te_mask.sum():,})")
        print(f"    CV-RMSE={mets['cv_rmse']:.2f}%  "
              f"R²={mets['r2']:.4f}  "
              f"sMAPE={mets['smape']:.2f}%  "
              f"MAE={mets['mae']:.4f}")

    # Overall = mean across folds; feeds the JSON summary.
    overall_metrics = {}
    if fold_results:
        overall_metrics = {k: float(np.mean([f[k] for f in fold_results]))
                            for k in ["cv_rmse", "r2", "smape", "mae", "rmse"]}

    if pooled_true:
        all_dates = np.concatenate(pooled_dates)
        all_true  = np.concatenate(pooled_true)
        all_pred  = np.concatenate(pooled_pred)

        # Sort chronologically and cap at FORECAST_DAY (plot_forecast applies +24h shift internally).
        order = np.argsort(all_dates)
        d_sorted, t_sorted, p_sorted = all_dates[order], all_true[order], all_pred[order]
        cap_mask = pd.to_datetime(d_sorted) < pd.Timestamp(FORECAST_DAY)
        plot_forecast(d_sorted[cap_mask], t_sorted[cap_mask], p_sorted[cap_mask],
                      f"Latest {HISTORY_HOURS // 24} Days", PLOT_DIR / "forecast_7day.png",
                      zoom_hours=HISTORY_HOURS)

    if fold_results:
        print(f"\n{'─'*62}")
        print("  WALK-FORWARD PER-FOLD BREAKDOWN")
        print(f"{'─'*62}")
        metrics_keys = ["cv_rmse", "r2", "smape", "mae", "rmse"]
        print(f"  {'Metric':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        print("  " + "─" * 46)
        for k in metrics_keys:
            vals = [f[k] for f in fold_results]
            print(f"  {k:<12} {np.mean(vals):>8.4f} {np.std(vals):>8.4f}"
                  f" {np.min(vals):>8.4f} {np.max(vals):>8.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # PRODUCTION MODEL — trained on ALL available data
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*62}")
    print("  PRODUCTION MODEL — trained on ALL available data")
    print(f"{'─'*62}")
    X_all = df[feat_cols].values
    y_all = df[TARGET].values
    w_all = make_weights(len(df))
    model_all = XGBRegressor(**PARAMS)
    model_all.fit(X_all, y_all, sample_weight=w_all)
    prod_path = OUT_DIR / "xgboost_production.joblib"
    joblib.dump(model_all, prod_path)
    print(f"  Trained on {len(df):,} rows (no holdout — this is the deploy model)")
    print(f"  Its performance is represented by the WALK-FORWARD PER-FOLD BREAKDOWN mean above,")
    print(f"  measured from the walk-forward retrained fold models, not this exact model.")
    print(f"  Saved → {prod_path.name}")

    # ══════════════════════════════════════════════════════════════════════
    # SHAP — feature importance/explainability for the production model
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*62}")
    print("  SHAP — feature importance (production model)")
    print(f"{'─'*62}")
    import shap
    shap_n = min(2000, len(df))
    X_shap = df[feat_cols].sample(n=shap_n, random_state=42)
    explainer = shap.TreeExplainer(model_all)
    shap_values = explainer.shap_values(X_shap)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap, show=False)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot → shap_summary.png  (sample n={shap_n})")

    shap_importance = dict(sorted(
        zip(feat_cols, np.abs(shap_values).mean(axis=0).tolist()),
        key=lambda kv: kv[1], reverse=True
    ))
    print("  Top 10 SHAP features:")
    for f, v in list(shap_importance.items())[:10]:
        print(f"    {f:<25} {v:.4f}")

    summary = {
        "model":            "XGBoost",
        "params":           PARAMS,
        "n_features":       len(feat_cols),
        "feature_cols":     feat_cols,
        "gap_hours":        GAP_HOURS,
        "summer_weight":    None,
        "eval_wf_folds":    fold_results,
        "eval_wf_mean":     {k: float(np.mean([f[k] for f in fold_results]))
                             for k in ["cv_rmse", "r2", "smape", "mae", "rmse"]}
                            if fold_results else {},
        "overall":          overall_metrics,
        "production_model": prod_path.name,  # file name only (no absolute path)
        "shap_importance":  shap_importance,
    }
    with open(OUT_DIR / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary → evaluation_summary.json")


if __name__ == "__main__":
    main()