"""
BiLSTM model — Office Water Consumption Forecasting
Sequence-to-one Bidirectional LSTM trained on sliding windows of features.
Inputs : features_selection/output/selected_features_data.csv
Outputs: output/bilstm_metrics.json, output/bilstm_test_predictions.csv,
         model/bilstm_model.pt, plots/bilstm_forecast_plot.png, plots/bilstm_loss_curve.png
Split  : chronological 70/15/15.
Run    : python train_bilstm.py
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import argparse
import os

BASE_DIR          = Path(__file__).resolve().parent
MAX_SEARCH_LEVELS = 5
OUTPUT_DIR_NAME   = "output"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
LOOKBACK      =     # hours of history fed as sequence
HIDDEN_SIZE   =     # right-sized for ~12k training samples
NUM_LAYERS    =     # 2 layers sufficient
DROPOUT       =     # between BiLSTM layers — regularises against memorisation
BATCH_SIZE    =     # larger batches → smoother gradient estimates
MAX_EPOCHS    =     # early stopping will terminate well before this
LR            =     # peak LR reached after warmup
LR_START      =     # initial LR at epoch 1 (warmup begins here)
LR_MIN        =     # floor LR at end of cosine decay
WARMUP_EPOCHS =     # epochs to ramp LR from LR_START to LR
ES_PATIENCE   =     # stop once val loss clearly plateaus
RANDOM_STATE  = 42
FORECAST_DAYS = 7

# Fixed display window for visual comparison (Monday-Sunday with typical office profile)
BEST_PLOT_WINDOW_START = pd.Timestamp("2017-11-27")
MASE_SEASONAL_PERIOD = 24  # hours — "same hour, previous day" naive benchmark
FUTURE_FORECAST_HOURS = 24  # length of the genuine forward forecast appended
                            # to the forecast plot, beyond the last known hour

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def _find_file(filename, override=None):
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
    sys.exit(1)


_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--input", default=None)
_parser.add_argument("--features-json", default=None)
_cli_args, _ = _parser.parse_known_args()

_input_override    = _cli_args.input or os.environ.get("LSTM_INPUT_FILE")
_features_override = _cli_args.features_json or os.environ.get("LSTM_FEATURES_JSON")

OUTPUT_DIR = BASE_DIR / OUTPUT_DIR_NAME
PLOT_DIR   = BASE_DIR / "plots"
MODEL_DIR  = BASE_DIR / "model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE    = _find_file("selected_features_data.csv", _input_override)
FEATURES_JSON = _find_file("selected_features.json",     _features_override)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def pbias(y_true, y_pred):
    """Percent Bias: 100 * sum(actual - pred) / sum(actual)."""
    denom = np.sum(y_true)
    if denom == 0:
        return np.nan
    return float(100 * np.sum(y_true - y_pred) / denom)


def seasonal_naive_mae(y_train, m=MASE_SEASONAL_PERIOD):
    """In-sample MAE of seasonal-naive forecast (y[t-m]) on training set."""
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) <= m:
        return np.nan
    return float(np.mean(np.abs(y_train[m:] - y_train[:-m])))


def mase(y_true, y_pred, naive_mae):
    """Mean Absolute Scaled Error (MASE < 1 beats seasonal-naive benchmark)."""
    if naive_mae == 0 or np.isnan(naive_mae):
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)) / naive_mae)


def seasonal_naive_mse(y_train, m=MASE_SEASONAL_PERIOD):
    """In-sample MSE of seasonal-naive forecast (y[t-m]) on training set."""
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) <= m:
        return np.nan
    return float(np.mean((y_train[m:] - y_train[:-m]) ** 2))


def rmsse(y_true, y_pred, naive_mse):
    """Root Mean Squared Scaled Error (RMSSE < 1 beats seasonal-naive benchmark)."""
    if naive_mse == 0 or np.isnan(naive_mse):
        return np.nan
    mse = np.mean((y_true - y_pred) ** 2)
    return float(np.sqrt(mse / naive_mse))


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
        "n":            int(len(y_true)),
        "MAE_L_per_h":  float(mean_absolute_error(y_true, y_pred)),
        "R2":           float(r2_score(y_true, y_pred)),
        "PBIAS_pct":    pbias(y_true, y_pred),
        "MASE":         mase(y_true, y_pred, naive_mae),
        "RMSSE":        rmsse(y_true, y_pred, naive_mse),
    }


def print_metrics_table(all_metrics):
    """Print a clean evaluation table with ratings for each metric."""
    keys  = ["n", "MAE_L_per_h", "R2", "PBIAS_pct", "MASE", "RMSSE"]
    col_w = 52
    print(f"\n{'Metric':<20}  {'Value':>10}  Rating")
    print("-" * col_w)
    for split, mdict in all_metrics.items():
        for k in keys:
            v       = mdict[k]
            val_str = f"{v:>10,.1f}" if k == "n" else f"{v:>10.4f}"
            rating  = f"  {RATERS[k](v)}" if k in RATERS else ""
            print(f"{k:<20}  {val_str}{rating}")
    print("-" * col_w)
    print("\nInterpretation (Moriasi et al. 2007 · Hyndman & Koehler 2006):")
    print("  PBIAS |<10%| Very good | |10-15%|   Good | |15-25%|  Satisfactory | >25%  Unsatisfactory")
    print("  MASE / RMSSE < 1.0 beats the seasonal-naive (same-hour-yesterday, "
          f"m={MASE_SEASONAL_PERIOD}h) forecast")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SequenceDataset(Dataset):
    """Sliding-window dataset: X = (lookback, n_features), y = scalar."""

    def __init__(self, X, y, lookback):
        self.X        = torch.tensor(X, dtype=torch.float32)
        self.y        = torch.tensor(y, dtype=torch.float32)
        self.lookback = lookback

    def __len__(self):
        return len(self.y) - self.lookback

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.lookback]   # (lookback, n_features)
        y_val = self.y[idx + self.lookback]          # scalar target
        return x_seq, y_val


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class BiLSTMForecaster(nn.Module):
    def __init__(self, n_features, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        eff_hidden = hidden_size * 2
        self.head = nn.Sequential(
            nn.Linear(eff_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# ---------------------------------------------------------------------------
# Forward forecast
# ---------------------------------------------------------------------------
def forecast_next_horizon(df, target_col, feature_cols, model, scaler, device,
                           lookback, horizon=FUTURE_FORECAST_HOURS):
    """Recursively forecast the `horizon` hours immediately after the last known timestamp in df."""
    dow_all  = df["datetime"].dt.dayofweek
    hour_all = df["datetime"].dt.hour
    door_lookup  = df.groupby([dow_all, hour_all])["door_open_prob"].first()
    door_default = df["door_open_prob"].mean()

    dt_hist = list(df["datetime"])
    y_hist  = list(df[target_col].values.astype(float))
    last_dt = dt_hist[-1]

    X_scaled_hist = scaler.transform(df[feature_cols].values.astype(float))
    seq_buffer = list(X_scaled_hist[-lookback:])

    model.eval()
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
        x_raw = np.array([[row[c] for c in feature_cols]], dtype=float)
        x_scaled = scaler.transform(x_raw)[0]
        seq_buffer.append(x_scaled)

        seq_input = np.array(seq_buffer[-lookback:], dtype=np.float32)
        X_tensor = torch.tensor(seq_input, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = float(model(X_tensor).cpu().numpy()[0])
        pred = max(pred, 0.0)

        out_rows.append({"datetime": t, "predicted_l": pred})
        dt_hist.append(t)
        y_hist.append(pred)

    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_loss_curve(train_losses, val_losses, lr_history, best_epoch, out_path):
    ep = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(ep, train_losses, label="Train Loss",  color="#1565C0", lw=2)
    ax1.plot(ep, val_losses,   label="Val Loss",    color="#E65100", lw=2)
    ax1.axvline(best_epoch, color="green", ls="--", alpha=0.6,
                label=f"Best Ep {best_epoch}")
    ax1.set_title("BiLSTM Training Curves  —  variance-normalized MSE\n"
                  "(each split divided by its own variance; both measured dropout OFF)",
                  fontweight="bold")
    ax1.set_xlabel("Number of Epoch")
    ax1.set_ylabel("Loss")
    # Clip top to the value after the first big drop (epoch 3 onward)
    if len(train_losses) > 3:
        y_top = max(max(train_losses[2:]), max(val_losses[2:])) * 1.15
        ax1.set_ylim(top=y_top)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(ep, lr_history, color="#2e7d32", lw=1.5, label="Learning Rate")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("LR")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved -> {out_path}")


def plot_forecast(datetimes, actual, pred, out_path, n_days=FORECAST_DAYS, future_df=None,
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

    dt = pd.Series(pd.to_datetime(datetimes))
    window_start = BEST_PLOT_WINDOW_START
    if not (dt.min() <= window_start <= dt.max()):
        window_start = dt.iloc[int(len(dt) * 0.25)].normalize()
    window_end   = window_start + pd.Timedelta(days=n_days)

    mask     = (dt >= window_start) & (dt < window_end)
    win_dt   = dt[mask]
    win_act  = np.asarray(actual)[mask.values]
    win_pred = np.asarray(pred)[mask.values]

    has_future = future_df is not None and len(future_df) > 0
    plot_end = future_df["datetime"].iloc[-1] + pd.Timedelta(hours=1) if has_future else window_end

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(win_dt, win_act,
            color=INK, linewidth=1.25, label="Actual", zorder=3,
            solid_capstyle="round")
    ax.plot(win_dt, win_pred,
            color=ACCENT, linewidth=1.25,
            label="BiLSTM forecast", zorder=2,
            solid_capstyle="round")

    if has_future:
        bridge_dt   = pd.concat([pd.Series(win_dt.values[-1:]), future_df["datetime"]])
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("  BiLSTM — Office Water Consumption Forecasting")
    print("=" * 65)
    print(f"  Device   : {device}")
    print(f"  Lookback : {LOOKBACK} hours")
    print(f"  Hidden   : {HIDDEN_SIZE} (x2 directions = {HIDDEN_SIZE*2})  Layers: {NUM_LAYERS}  Dropout: {DROPOUT}")
    print(f"\n  Loss-curve design active:")
    print(f"    [1] Train loss recomputed post-epoch in eval mode (dropout OFF)")
    print(f"    [2] Each split normalized by its own variance")

    print(f"\nLoading data from     : {INPUT_FILE}")
    print(f"Loading features from : {FEATURES_JSON}")

    df = pd.read_csv(INPUT_FILE, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    with open(FEATURES_JSON) as f:
        feat_summary = json.load(f)
    feature_cols = feat_summary["selected_features"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    target_col = "consumption_l"

    n         = len(df)
    train_end = int(n * TRAIN_FRAC)
    val_end   = int(n * (TRAIN_FRAC + VAL_FRAC))

    train_df = df.iloc[:train_end]
    val_df   = df.iloc[train_end:val_end]
    test_df  = df.iloc[val_end:]

    print(f"\nFeatures used ({len(feature_cols)}): {feature_cols}")
    print(f"Split sizes -> train: {len(train_df):,} | val: {len(val_df):,} | "
          f"test: {len(test_df):,}")
    print(f"Test  period: {test_df['datetime'].min()} -> {test_df['datetime'].max()}")

    # ---- Scale features (fit only on train) --------------------------------
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(train_df[feature_cols].values)
    X_val_sc   = scaler.transform(val_df[feature_cols].values)
    X_test_sc  = scaler.transform(test_df[feature_cols].values)

    y_train = train_df[target_col].values.astype(np.float32)
    y_val   = val_df[target_col].values.astype(np.float32)
    y_test  = test_df[target_col].values.astype(np.float32)

    # Normalize loss by split variance for fair train-vs-val comparison
    var_train = float(np.var(y_train))
    var_val   = float(np.var(y_val))
    print(f"\n  var_train={var_train:.2f}  var_val={var_val:.2f}  "
          f"(each loss normalized by its own split variance)")

    # ---- Datasets & loaders --------------------------------------------------
    train_ds = SequenceDataset(X_train_sc, y_train, LOOKBACK)
    val_ds   = SequenceDataset(X_val_sc,   y_val,   LOOKBACK)
    test_ds  = SequenceDataset(X_test_sc,  y_test,  LOOKBACK)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # ---- Model ---------------------------------------------------------------
    n_features = len(feature_cols)
    model = BiLSTMForecaster(n_features, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR_START, weight_decay=1e-4)

    def lr_lambda(epoch):
        """Warmup for WARMUP_EPOCHS then cosine decay to LR_MIN."""
        if epoch < WARMUP_EPOCHS:
            return (LR_START + (LR - LR_START) * (epoch + 1) / WARMUP_EPOCHS) / LR_START
        progress = (epoch - WARMUP_EPOCHS) / max(1, MAX_EPOCHS - WARMUP_EPOCHS)
        cosine   = 0.5 * (1 + math.cos(math.pi * progress))
        return (LR_MIN + (LR - LR_MIN) * cosine) / LR_START

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)

    # ---- Training loop --------------------------------------------------------
    print(f"\nTraining BiLSTM (max {MAX_EPOCHS} epochs, "
          f"early stopping patience={ES_PATIENCE})...")
    train_losses, val_losses, lr_history = [], [], []
    best_val_loss  = float("inf")
    best_state     = None
    best_epoch     = 0
    epochs_no_imp  = 0

    for epoch in range(1, MAX_EPOCHS + 1):

        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimiser.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

        # Train loss recomputed post-update in eval mode (dropout OFF)
        model.eval()
        train_loss = 0.0
        with torch.no_grad():
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred = model(Xb)
                train_loss += criterion(pred, yb).item() * len(yb)
        train_loss /= len(train_ds)

        # -- val: eval mode, dropout OFF ---------------------------------------
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred = model(Xb)
                val_loss += criterion(pred, yb).item() * len(yb)
        val_loss /= len(val_ds)

        # Normalize each split by its own variance for a fair comparison
        train_losses.append(train_loss / var_train)
        val_losses.append(val_loss   / var_val)

        scheduler.step()
        lr_history.append(optimiser.param_groups[0]["lr"])

        # Early stopping tracks the raw (un-normalized) val MSE
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch    = epoch
            epochs_no_imp = 0
        else:
            epochs_no_imp += 1

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimiser.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{MAX_EPOCHS} | "
                  f"train MSE: {train_loss:8.2f} | "
                  f"val MSE: {val_loss:8.2f} | "
                  f"lr: {lr_now:.2e} | "
                  f"no-imp: {epochs_no_imp}/{ES_PATIENCE}")

        if epochs_no_imp >= ES_PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(best val MSE={best_val_loss:.2f} at epoch {best_epoch})")
            break

    # ---- Restore best weights --------------------------------------------------
    model.load_state_dict(best_state)
    print(f"\nBest epoch: {best_epoch}  |  Best val MSE: {best_val_loss:.4f}")

    # ---- Test predictions --------------------------------------------------------
    model.eval()
    preds = []
    with torch.no_grad():
        for Xb, _ in test_loader:
            preds.append(model(Xb.to(device)).cpu().numpy())
    pred_l = np.clip(np.concatenate(preds), 0, None)

    # The dataset drops the first LOOKBACK rows (no history yet), so align targets
    actual_l       = y_test[LOOKBACK:]
    test_datetimes = test_df["datetime"].values[LOOKBACK:]

    naive_mae = seasonal_naive_mae(y_train, m=MASE_SEASONAL_PERIOD)
    naive_mse = seasonal_naive_mse(y_train, m=MASE_SEASONAL_PERIOD)
    print(f"\nMASE/RMSSE seasonal-naive scale (m={MASE_SEASONAL_PERIOD}h): "
          f"in-sample MAE={naive_mae:.4f} L/h, MSE={naive_mse:.4f} (L/h)^2")

    all_metrics = {"test": compute_metrics(actual_l, pred_l, naive_mae, naive_mse)}

    print("\n" + "=" * 65)
    print("  Evaluation (litres/hour)")
    print("=" * 65)
    print_metrics_table(all_metrics)

    # ---- Save predictions CSV ---------------------------------------------------
    test_out = pd.DataFrame({
        "datetime":      test_datetimes,
        "consumption_l": actual_l,
        "predicted_l":   pred_l,
        "abs_error_l":   np.abs(actual_l - pred_l),
    })
    csv_path = OUTPUT_DIR / "bilstm_test_predictions.csv"
    test_out.to_csv(csv_path, index=False)
    print(f"\nSaved -> {csv_path}")

    # ---- Save metrics JSON -------------------------------------------------------
    metrics_json = {}
    for split, split_metrics in all_metrics.items():
        clean = {mk: (None if isinstance(mv, float) and np.isnan(mv) else mv)
                 for mk, mv in split_metrics.items()}
        clean["ratings"] = {mk: RATERS[mk](split_metrics[mk]) for mk in RATERS}
        metrics_json[split] = clean
    metrics_json["mase_naive_scale_L_per_h"] = naive_mae
    metrics_json["rmsse_naive_scale_L_per_h2"] = naive_mse
    json_path = OUTPUT_DIR / "bilstm_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics_json, f, indent=4)
    print(f"Saved -> {json_path}")

    # ---- Save model --------------------------------------------------------------
    model_path = MODEL_DIR / "bilstm_model.pt"
    torch.save({
        "state_dict":   best_state,
        "feature_cols": feature_cols,
        "scaler_mean":  scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "hyperparams": {
            "lookback":     LOOKBACK,
            "hidden_size":  HIDDEN_SIZE,
            "num_layers":   NUM_LAYERS,
            "dropout":      DROPOUT,
            "n_features":   n_features,
            "bidirectional": True,
        },
        "best_epoch":   best_epoch,
        "best_val_mse": best_val_loss,
    }, model_path)
    print(f"Saved -> {model_path}")

    # ---- Loss curve plot -----------------------------------------------------------
    print("\nGenerating loss curve plot...")
    plot_loss_curve(train_losses, val_losses, lr_history, best_epoch,
                    PLOT_DIR / "bilstm_loss_curve.png")

    # ---- Forward forecast beyond sample window ----
    test_dt_series = pd.Series(pd.to_datetime(test_datetimes))
    _plot_window_start = BEST_PLOT_WINDOW_START
    if not (test_dt_series.min() <= _plot_window_start <= test_dt_series.max()):
        _plot_window_start = test_dt_series.iloc[int(len(test_dt_series) * 0.25)].normalize()
    forecast_anchor = _plot_window_start + pd.Timedelta(days=FORECAST_DAYS)
    history_df = df[df["datetime"] < forecast_anchor].reset_index(drop=True)
    print(f"\nForecasting the next {FUTURE_FORECAST_HOURS}h beyond "
          f"{history_df['datetime'].iloc[-1]} (pretending nothing after that "
          f"point is known)...")
    future_df = forecast_next_horizon(
        history_df, target_col, feature_cols, model, scaler, device, LOOKBACK,
        horizon=FUTURE_FORECAST_HOURS,
    )
    print(future_df.to_string(index=False))

    # ---- Forecast plot --------------------------------------------------------------
    print("Generating forecast plot...")
    error_scale = float(np.mean(np.abs(actual_l - pred_l)))  # test-set MAE
    plot_forecast(
        test_datetimes, actual_l, pred_l,
        PLOT_DIR / "bilstm_forecast_plot.png",
        future_df=future_df,
        error_scale=error_scale,
    )

    print("\nDone.")
    return model, all_metrics


if __name__ == "__main__":
    run()