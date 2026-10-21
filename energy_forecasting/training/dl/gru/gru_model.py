"""
gru_model.py
============
Bidirectional GRU neural network for 24-hour ahead energy consumption forecasting.

Architecture & Training:
- 2-layer Bidirectional GRU (hidden_size=64) with residual skip to the last input step.
- Augmented input tensor: feature matrix + normalized target history + genuine data flag.
- Block-cyclic validation split with temporal embargo to prevent lookahead leakage.
- Trained using AdamW optimizer with Huber loss and cosine annealing learning rate schedule.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import warnings
warnings.filterwarnings("ignore")
import json
import math
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from training.ml.config import (
    FEATURES_CSV, TARGET, GAP_HOURS, THRESHOLDS,
    SELECTED_FEATURES, FEATURE_SELECTION_JSON,
    FORECAST_DAY, HISTORY_HOURS,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print(f"  Using CUDA : {torch.cuda.get_device_name(0)}")
else:
    print("  CUDA not available — using CPU")

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_COLOR = "#00897B"

# ═══════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

LOOKBACK           = 168
SEASONAL_LAG       = 168
HIDDEN_SIZE        = 64
NUM_LAYERS         = 2
BATCH_SIZE         = 48
LR                 = 1e-4
WARMUP_EPOCHS      = 2
EPOCHS             = 50
COSINE_EPOCHS      = 50
MIN_EPOCHS         = 8
HUBER_DELTA        = 0.5
N_EXTRA_CHANNELS   = 2

# Regularization & early stopping
USE_DROPOUT        = False
DROPOUT_RATE       = 0.2
WEIGHT_DECAY       = 0.0
EARLY_STOPPING     = True
PATIENCE           = 10
USE_EMA            = False
EMA_DECAY          = 0.999

# Block-cyclic validation split configuration
BLOCK_HOURS         = 168
DEV_CYCLE           = ["train"] * 13 + ["val"] * 3
BLOCK_EMBARGO_HOURS = LOOKBACK

# ═══════════════════════════════════════════════════════════════════════════
# EMA (kept for completeness, unused by default)
# ═══════════════════════════════════════════════════════════════════════════

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.num_updates = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.backup = {}

    def update(self):
        self.num_updates += 1
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        with torch.no_grad():
            for name, param in self.model.state_dict().items():
                self.shadow[name] = (d * self.shadow[name] + (1 - d) * param.detach()).clone()

    def apply_shadow(self):
        self.backup = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(self.shadow, strict=True)

    def restore(self):
        self.model.load_state_dict(self.backup, strict=True)
        self.backup = {}

# ═══════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, genuine, lookback, y_mean, y_std,
                 seasonal_lag=SEASONAL_LAG, eligible=None):
        assert lookback >= seasonal_lag
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(np.nan_to_num(y, nan=0.0))
        self.lookback = lookback
        self.seasonal_lag = seasonal_lag
        self.y_mean, self.y_std = float(y_mean), float(y_std) + 1e-9

        g = np.asarray(genuine, dtype=bool)
        g_f = torch.FloatTensor(g.astype(np.float32))
        self.z_hist = ((self.y - self.y_mean) / self.y_std) * g_f
        self.g_flag = g_f

        idx_all = np.arange(len(y) - lookback)
        t = idx_all + lookback
        base_ok = g[t] & g[t - seasonal_lag]
        if eligible is not None:
            elig = np.asarray(eligible, dtype=bool)
            base_ok = base_ok & elig[t]
        self.valid_idx = idx_all[base_ok]

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, i):
        idx = int(self.valid_idx[i])
        t = idx + self.lookback
        z = (self.y[t] - self.y_mean) / self.y_std
        baseline = self.y[t - self.seasonal_lag]
        window = torch.cat([self.X[idx:t],
                            self.z_hist[idx:t].unsqueeze(-1),
                            self.g_flag[idx:t].unsqueeze(-1)], dim=-1)
        return window, z, baseline

# ═══════════════════════════════════════════════════════════════════════════
# BLOCK-CYCLIC SPLIT BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def build_dev_split_masks(dev_n, block_hours, cycle, embargo_hours):
    n_blocks = math.ceil(dev_n / block_hours)
    block_label = [cycle[i % len(cycle)] for i in range(n_blocks)]
    hour_label = np.empty(dev_n, dtype=object)
    for b, lbl in enumerate(block_label):
        s = b * block_hours
        e = min(dev_n, s + block_hours)
        hour_label[s:e] = lbl

    labels = sorted(set(cycle))
    eligible = {lbl: np.zeros(dev_n, dtype=bool) for lbl in labels}

    i = 0
    while i < dev_n:
        j = i
        while j < dev_n and hour_label[j] == hour_label[i]:
            j += 1
        lbl = hour_label[i]
        elig_start = min(i + embargo_hours, j)
        eligible[lbl][elig_start:j] = True
        i = j

    return eligible

# ═══════════════════════════════════════════════════════════════════════════
# MODEL: GRU (dropout parameterized but OFF by default)
# ═══════════════════════════════════════════════════════════════════════════

class GRUModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 48, num_layers: int = 2,
                 use_dropout: bool = False, dropout_rate: float = 0.2):
        super().__init__()

        gru_dropout = dropout_rate if (use_dropout and num_layers > 1) else 0.0

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout,
        )
        eff_hidden = hidden_size * 2

        self.drop_context = nn.Dropout(dropout_rate) if use_dropout else nn.Identity()
        self.drop_fc1 = nn.Dropout(dropout_rate) if use_dropout else nn.Identity()
        self.drop_fc2 = nn.Dropout(dropout_rate * 0.5) if use_dropout else nn.Identity()

        self.fc1 = nn.Linear(eff_hidden + input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last_input = x[:, -1, :]

        gru_out, _ = self.gru(x)
        context = gru_out[:, -1, :]
        context = self.drop_context(context)

        combined = torch.cat([context, last_input], dim=1)

        out = self.relu(self.fc1(combined))
        out = self.drop_fc1(out)
        out = self.relu(self.fc2(out))
        out = self.drop_fc2(out)
        return self.fc3(out)

# ═══════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_model(model, train_loader, val_loader, epochs, patience,
                min_epochs=MIN_EPOCHS, early_stopping=EARLY_STOPPING,
                use_ema=USE_EMA, ema_decay=EMA_DECAY):
    model = model.to(DEVICE)
    criterion = nn.HuberLoss(delta=HUBER_DELTA, reduction='none')

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    ema = EMA(model, decay=ema_decay) if use_ema else None

    steps_per_epoch = len(train_loader)
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    cosine_steps = COSINE_EPOCHS * steps_per_epoch

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return (current_step + 1) / warmup_steps
        progress = min(
            1.0,
            (current_step - warmup_steps) / max(1, (cosine_steps - warmup_steps))
        )
        eta_min_ratio = 1e-6 / LR
        return eta_min_ratio + (1 - eta_min_ratio) * 0.5 * (
            1 + math.cos(math.pi * progress)
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val, best_epoch, pc = float('inf'), 0, 0
    best_state = None
    train_losses, val_losses, learning_rates, gaps = [], [], [], []

    for epoch in range(epochs):
        model.train()
        for Xb, zb, _ in train_loader:
            Xb, zb = Xb.to(DEVICE), zb.to(DEVICE)

            optimizer.zero_grad()
            loss = criterion(model(Xb).squeeze(-1), zb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            if use_ema:
                ema.update()

        model.eval()

        train_loss = 0.0
        with torch.no_grad():
            for Xb, zb, _ in train_loader:
                Xb, zb = Xb.to(DEVICE), zb.to(DEVICE)
                train_loss += criterion(model(Xb).squeeze(-1), zb).mean().item() * Xb.size(0)
        train_loss /= len(train_loader.dataset)

        if use_ema:
            ema.apply_shadow()

        val_loss = 0.0
        with torch.no_grad():
            for Xb, zb, _ in val_loader:
                Xb, zb = Xb.to(DEVICE), zb.to(DEVICE)
                val_loss += criterion(model(Xb).squeeze(-1), zb).mean().item() * Xb.size(0)
        val_loss /= len(val_loader.dataset)

        ema_snapshot = None
        if use_ema:
            ema_snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            ema.restore()

        gap = val_loss - train_loss
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        gaps.append(gap)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            gap_tag = "✅ val>train" if gap > 0 else "⚠️  val<train"
            print(f"  Epoch {epoch+1:3d}/{epochs} | Train: {train_loss:.6f} | "
                  f"Val: {val_loss:.6f} | Gap: {gap:+.6f} {gap_tag} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val and (epoch + 1) >= min_epochs:
            best_val, best_epoch, pc = val_loss, epoch + 1, 0
            best_state = ema_snapshot if use_ema else \
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"    ✓ new best (epoch {best_epoch}, val {val_loss:.6f})")
        elif (epoch + 1) < min_epochs:
            if epoch == min_epochs - 2:
                best_state = ema_snapshot if use_ema else \
                    {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch + 1
        else:
            pc += 1
            if early_stopping and pc >= patience:
                print(f"  Early stopping at epoch {epoch+1} (best val epoch {best_epoch})")
                break

    if best_state is None:
        raise RuntimeError("No best model captured during training.")

    print(f"\n  Training complete: ran {len(train_losses)} epochs")
    print(f"  Best val loss: {best_val:.6f}  (epoch {best_epoch})")

    below_epoch = next((i + 1 for i, g in enumerate(gaps) if g < 0), None)
    if below_epoch:
        print(f"  ℹ️  Val dipped below train at epoch {below_epoch} (check trend).")
    else:
        print(f"  ✅ Val stayed at/above train for all {len(train_losses)} epochs.")

    model.load_state_dict(best_state)
    return model, train_losses, val_losses, learning_rates, best_epoch, gaps

# ═══════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, y_train=None, seasonal_period=168):
    y_pred = np.clip(y_pred, 0, None)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    if y_train is not None and len(y_train) > seasonal_period:
        naive_errors = np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period])
        mase = mae / (float(np.mean(naive_errors)) + 1e-9)
    else:
        mase = None
    nmbe = float(np.sum(y_true - y_pred) / (np.mean(y_true) * len(y_true) + 1e-9) * 100)
    return dict(
        r2=float(r2_score(y_true, y_pred)),
        rmse=rmse,
        mae=mae,
        smape=float(np.mean(
            2 * np.abs(y_pred - y_true) /
            (np.abs(y_true) + np.abs(y_pred) + 1e-8)
        ) * 100),
        cv_rmse=rmse / (np.mean(y_true) + 1e-9) * 100,
        nmbe=nmbe,
        mase=float(mase) if mase is not None else None,
    )

def _tag(metric, value):
    th = THRESHOLDS.get(metric)
    if th is None:
        return ""
    if metric == "r2":
        if value >= th["excellent"]: return "✅ excellent"
        if value >= th["good"]: return "✅ good"
        return f"⚠️  below {th['good']}"
    if value <= th["excellent"]: return "✅ excellent"
    if value <= th["good"]: return "✅ good"
    return "⚠️  above threshold"

def print_metrics(m, label=""):
    tag = f"  [{label}]" if label else ""
    print(f"\n{'='*62}\n  RESULTS{tag}\n{'='*62}")
    print(f"  CV-RMSE  = {m['cv_rmse']:6.2f}%    {_tag('cv_rmse', m['cv_rmse'])}")
    print(f"  R²       = {m['r2']:6.4f}     {_tag('r2', m['r2'])}")
    print(f"  sMAPE    = {m['smape']:6.2f}%    {_tag('smape', m['smape'])}")
    print(f"  MAE      = {m['mae']:6.4f} kWh")
    print(f"  RMSE     = {m['rmse']:6.4f} kWh")
    print(f"  NMBE     = {m['nmbe']:+.2f}%  "
          f"{'✅ pass' if abs(m['nmbe']) <= 10 else '⚠️  >±10% (ASHRAE)'}")
    if m.get('mase') is not None:
        print(f"  MASE     = {m['mase']:.4f}  "
              f"{'✅ <1.0 (beats naive)' if m['mase'] < 1.0 else '⚠️  ≥1.0 (worse than naive)'}")
    print(f"{'='*62}")

def plot_training_curves(train_losses, val_losses, learning_rates, best_epoch,
                         out_path, gaps=None):
    ep = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    ax1.plot(ep, train_losses, label='Training Loss', color='#1565C0', lw=2)
    ax1.plot(ep, val_losses, label='Validation Loss', color='#E65100', lw=2)
    ax1.axvline(best_epoch, color='green', ls='--', alpha=0.6,
                label=f'Best Val Checkpoint (Epoch {best_epoch})')

    ax1.set_xlabel('Number of Epoch')
    ax1.set_ylabel('Loss')
    reg_str = "Zero Regularization" if not USE_DROPOUT and WEIGHT_DECAY == 0 \
              else f"Dropout={DROPOUT_RATE if USE_DROPOUT else 0}, WD={WEIGHT_DECAY}"
    ax1.set_title(f'GRU — {reg_str}', fontweight='bold')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(alpha=0.3)
    ax1.set_ylim(bottom=0)

    ax2.plot(ep, learning_rates, color=MODEL_COLOR, lw=2, label='Learning Rate')
    ax2.set_xlabel('Number of Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule (Warmup + Cosine Annealing)',
                  fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend(loc='upper right')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Training curves → {out_path.name}")


def predict_calendar_window(df, X_all, y_all, genuine, model,
                            forecast_day, history_hours, lookback,
                            y_mean, y_std, n_extra_channels=N_EXTRA_CHANNELS):
    """
    Generate model predictions over a specified calendar window.
    Constructs input windows mirroring TimeSeriesDataset (features + z_hist + genuine flag).
    """
    g = genuine.astype(np.float32)
    # Zero-fill missing values for model inputs to prevent NaN propagation
    y_clean = np.nan_to_num(y_all, nan=0.0)
    z_hist_all = ((y_clean - y_mean) / y_std) * g

    raw_start = pd.Timestamp(forecast_day) - pd.Timedelta(hours=history_hours + 24)
    raw_end = pd.Timestamp(forecast_day) - pd.Timedelta(hours=1)
    dt_all = df["datetime"].values
    idx = np.where((dt_all >= np.datetime64(raw_start)) & (dt_all <= np.datetime64(raw_end)))[0]
    idx = idx[idx >= lookback]

    windows = [
        np.concatenate([
            X_all[t - lookback:t],
            z_hist_all[t - lookback:t, None],
            g[t - lookback:t, None],
        ], axis=-1)
        for t in idx
    ]
    X_win = torch.FloatTensor(np.stack(windows)).to(DEVICE)

    model.eval()
    with torch.no_grad():
        z_hat = model(X_win).squeeze(-1).cpu().numpy()
    pred = z_hat * y_std + y_mean
    # Mask non-genuine/imputed intervals to NaN for clean plotting of actual readings
    actual = y_clean[idx].copy()
    actual[genuine[idx] == False] = np.nan
    dates = dt_all[idx]
    return dates, actual, pred


def plot_forecast(dates_te, y_te, y_pred, label, out_path, zoom_hours=168):
    """Plot actual vs. predicted consumption curves with confidence intervals."""
    INK    = "#0b0b0b"   # Actual readings
    ACCENT = "#2a78d6"   # Model forecast
    MUTED  = "#898781"   # Axis labels and ticks
    GRID   = "#e1e0d9"   # Background gridlines

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

    dates_te = pd.to_datetime(dates_te)
    dates_te = dates_te + pd.Timedelta(hours=24)

    y_pred = np.clip(y_pred, 0, None)
    n = len(y_te)
    he = n - 24
    hs = max(0, he - zoom_hours)
    dh, ah, ph = dates_te[hs:he+1], y_te[hs:he+1], y_pred[hs:he+1]
    df_fore, pf = dates_te[he:he+25], y_pred[he:he+25]
    # Compute residual standard error excluding NaN gaps
    rs = np.nanstd(ah - ph)
    ci_up = pf + 1.645 * rs
    ci_lo = np.clip(pf - 1.645 * rs, 0, None)

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.fill_between(df_fore, ci_lo, ci_up,
                    color=ACCENT, alpha=0.15, linewidth=0, zorder=1,
                    label="forecast interval")
    ax.plot(dh, ah, color=INK, linewidth=1.25, zorder=3,
            solid_capstyle="round", label="Actual")
    ax.plot(dh, ph, color=ACCENT, linewidth=1.25, zorder=2,
            solid_capstyle="round", label="BiGRU forecast")
    ax.plot(df_fore, pf, color=ACCENT, linewidth=1.25, zorder=2,
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
        loc="upper right", frameon=True, framealpha=0.9,
        edgecolor=GRID, facecolor="white",
        fontsize=9, handlelength=2.2,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Plot → {out_path.name}")

    # Reset matplotlib rcParams to defaults
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
    print("  GRU RUN — PRODUCTION (Zero-Reg, Ablation-Validated)")
    print(f"  Hidden: {HIDDEN_SIZE}×2 | Layers: {NUM_LAYERS} | "
          f"Extra channels: {N_EXTRA_CHANNELS}")
    print(f"  Dropout: {USE_DROPOUT} | Weight decay: {WEIGHT_DECAY} | "
          f"Early stopping: {EARLY_STOPPING} (patience={PATIENCE}) | EMA: {USE_EMA}")
    print(f"  LR: {LR:.2e} (warmup {WARMUP_EPOCHS}, cosine {COSINE_EPOCHS}) | "
          f"Batch: {BATCH_SIZE} | Huber δ: {HUBER_DELTA}")
    print("=" * 62)

    df = pd.read_csv(FEATURES_CSV)
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
        raise FileNotFoundError(f"feature_selection.json missing: {FEATURE_SELECTION_JSON}")
    feat_cols = [c for c in SELECTED_FEATURES if c in df.columns]
    if not feat_cols:
        raise ValueError("No selected features found in CSV.")

    n_before = len(df)
    df = df.drop_duplicates(subset="datetime").set_index("datetime")
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    genuine = pd.Series(True, index=df.index).reindex(full_idx, fill_value=False)
    df = df.reindex(full_idx)
    print(f"  Hourly reindex: {n_before:,} → {len(df):,} "
          f"({int((~genuine).sum()):,} gap rows inserted)")
    df[feat_cols] = df[feat_cols].interpolate(method="time", limit=24,
                                              limit_direction="both")
    train_end_for_median = int(len(df) * 0.65)
    train_median = df.iloc[:train_end_for_median][feat_cols].median()
    df[feat_cols] = df[feat_cols].fillna(train_median)
    df = df.reset_index().rename(columns={"index": "datetime"})
    genuine = genuine.values

    N = len(df)
    val_end    = int(N * 0.80)
    test_start = val_end + GAP_HOURS
    sl_te = slice(test_start, N)
    DEV_N = val_end

    print(f"  Test split (chronological): [{test_start}:{N}] "
          f"({N - test_start:,} hours, gap={GAP_HOURS}h)")
    print(f"  Dev pool (blocked/cyclic): [0:{DEV_N}] ({DEV_N:,} hours)")

    eligible_dev = build_dev_split_masks(
        DEV_N, BLOCK_HOURS, DEV_CYCLE, BLOCK_EMBARGO_HOURS
    )
    eligible_train = eligible_dev["train"]
    eligible_val   = eligible_dev["val"]

    feat_matrix_raw = df[feat_cols].values
    g_dev = genuine[:DEV_N]
    train_fit_mask = g_dev & eligible_train
    scaler = RobustScaler()
    scaler.fit(feat_matrix_raw[:DEV_N][train_fit_mask])
    X_all = scaler.transform(feat_matrix_raw)
    y_all = df[TARGET].values.astype(float)

    X_dev, y_dev = X_all[:DEV_N], y_all[:DEV_N]
    X_te,  y_te,  g_te  = X_all[sl_te], y_all[sl_te], genuine[sl_te]

    y_tr_genuine  = y_dev[g_dev & eligible_train]
    y_val_genuine = y_dev[g_dev & eligible_val]
    y_mean, y_std = float(np.nanmean(y_tr_genuine)), float(np.nanstd(y_tr_genuine))
    print(f"\n  Target scaling (train): mean={y_mean:.4f}  std={y_std:.4f} kWh")
    print(f"  Train CV={y_tr_genuine.std()/y_tr_genuine.mean()*100:.1f}% | "
          f"Val CV={y_val_genuine.std()/y_val_genuine.mean()*100:.1f}%")

    ds_kw = dict(y_mean=y_mean, y_std=y_std, seasonal_lag=SEASONAL_LAG)
    train_ds = TimeSeriesDataset(X_dev, y_dev, g_dev, LOOKBACK, eligible=eligible_train, **ds_kw)
    val_ds   = TimeSeriesDataset(X_dev, y_dev, g_dev, LOOKBACK, eligible=eligible_val,   **ds_kw)
    test_ds  = TimeSeriesDataset(X_te,  y_te,  g_te,  LOOKBACK, **ds_kw)
    print(f"  Samples: train={len(train_ds):,} val={len(val_ds):,} test={len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = GRUModel(len(feat_cols) + N_EXTRA_CHANNELS, HIDDEN_SIZE, NUM_LAYERS,
                      use_dropout=USE_DROPOUT, dropout_rate=DROPOUT_RATE)

    model, train_losses, val_losses, lrs, best_epoch, gaps = train_model(
        model, train_loader, val_loader, EPOCHS, PATIENCE
    )

    plot_training_curves(train_losses, val_losses, lrs, best_epoch,
                         OUT_DIR / "training_curves.png", gaps=gaps)

    loss_history = {
        "model": "GRU-production-zero-reg",
        "train_losses": train_losses,
        "val_losses": val_losses,
        "learning_rates": lrs,
        "gaps": gaps,
        "best_epoch": best_epoch,
        "use_dropout": USE_DROPOUT,
        "weight_decay": WEIGHT_DECAY,
        "early_stopping": EARLY_STOPPING,
        "use_ema": USE_EMA,
        "ablation_note": "dropout=0.2+wd=1e-4 tested and rejected — see docstring",
    }
    with open(OUT_DIR / "loss_history.json", "w") as f:
        json.dump(loss_history, f, indent=2)
    print(f"  Loss history → loss_history.json")

    model.eval()
    preds, actuals, baselines = [], [], []
    with torch.no_grad():
        for Xb, zb, bb in test_loader:
            Xb = Xb.to(DEVICE)
            z_hat = model(Xb).squeeze(-1).cpu().numpy()
            if z_hat.ndim == 0:
                z_hat = z_hat.reshape(1)
            preds.extend((z_hat * y_std + y_mean).tolist())
            actuals.extend((zb.numpy() * y_std + y_mean).tolist())
            baselines.extend(bb.numpy().tolist())

    pf, af, bf = np.array(preds), np.array(actuals), np.array(baselines)

    naive_mets = compute_metrics(af, bf, y_train=y_tr_genuine)
    print(f"\n  Seasonal-naive reference:")
    print(f"    R²={naive_mets['r2']:.4f}  CV-RMSE={naive_mets['cv_rmse']:.2f}%"
          f"  MASE={naive_mets['mase']:.3f}")

    mets = compute_metrics(af, pf, y_train=y_tr_genuine)
    print_metrics(mets, label="Test (GRU, production)")
    improve = (naive_mets['rmse'] - mets['rmse']) / naive_mets['rmse'] * 100
    improve_mae = (naive_mets['mae'] - mets['mae']) / naive_mets['mae'] * 100
    beat = "✅ beats naive" if improve > 0 else "⚠️  WORSE than t-168 naive"
    print(f"  RMSE vs seasonal naive (test): {improve:+.1f}%  {beat}")
    print(f"  MAE  vs seasonal naive (test): {improve_mae:+.1f}%")

    # Diagnostic plot over the configured calendar window
    win_dates, win_actual, win_pred = predict_calendar_window(
        df, X_all, y_all, genuine, model,
        forecast_day=FORECAST_DAY, history_hours=HISTORY_HOURS,
        lookback=LOOKBACK, y_mean=y_mean, y_std=y_std,
    )
    plot_forecast(win_dates, win_actual, win_pred,
                  "Test", OUT_DIR / "forecast_test.png", zoom_hours=HISTORY_HOURS)

    summary = {
        "model": "GRU-production-zero-reg",
        "params": {
            "lookback": LOOKBACK, "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS, "batch_size": BATCH_SIZE,
            "lr": LR, "warmup_epochs": WARMUP_EPOCHS,
            "cosine_epochs": COSINE_EPOCHS,
            "n_extra_channels": N_EXTRA_CHANNELS,
            "huber_delta": HUBER_DELTA,
            "epochs": EPOCHS, "patience": PATIENCE,
            "use_dropout": USE_DROPOUT,
            "weight_decay": WEIGHT_DECAY,
            "early_stopping": EARLY_STOPPING,
            "use_ema": USE_EMA,
        },
        "input_size": "len(feat_cols) + n_extra_channels",
        "architecture": [
            f"Bidirectional_GRU({NUM_LAYERS} layer, no dropout)",
            "TargetHistory+GenuineFlag input channels",
            "LastHidden+LastInput skip",
            f"MLP({HIDDEN_SIZE*2}+F+{N_EXTRA_CHANNELS} -> 64 -> 32 -> 1)",
            "RobustScaler",
        ],
        "regularization": "None (zero-regularization validated on dev split)",
        "split_method": "Test=final chronological 20%. Train/Val=blocked "
                        "cyclic weekly assignment within preceding 80% dev pool.",
        "target_scaling": {"mean": y_mean, "std": y_std},
        "test_metrics": mets,
        "naive_reference_metrics": naive_mets,
        "device": str(DEVICE),
    }
    with open(OUT_DIR / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary → evaluation_summary.json")

    final_model_path = OUT_DIR / "gru_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_class": "GRUModel",
        "model_init_args": {
            "input_size": len(feat_cols) + N_EXTRA_CHANNELS,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "use_dropout": USE_DROPOUT,
            "dropout_rate": DROPOUT_RATE,
        },
        "n_extra_channels": N_EXTRA_CHANNELS,
        "feature_columns": feat_cols,
        "scaler": scaler,
        "target_scaling": {"mean": y_mean, "std": y_std},
        "hyperparams": summary["params"],
        "test_metrics": mets,
        "device": str(DEVICE),
    }, final_model_path)
    print(f"  Final model bundle → {final_model_path.name}")
    print("  DONE")

if __name__ == "__main__":
    main()