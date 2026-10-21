"""
latent_sweep.py — Latent Dimension Sweep (Window=24h fixed)
=============================================================
Tests latent dimensions: [2, 4, 6, 8, 12, 16]
Window fixed at 24h — the literature-standard daily window.

For each latent dim:
  - Same two-path ETL (train on clean, test on raw)
  - Same architecture (hidden=32, 2-layer enc/dec)
  - Same training settings (Adam, lr=1e-3, patience=10)
  - Threshold: 95th pct of train errors

OUTPUTS:
    latent_sweep/results.csv
    latent_sweep/plots/
"""

import os, sys, warnings, pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import RAW_DATA_CSV as RAW_CSV, LATENT_SWEEP_DIR as OUT_DIR

# ── CONFIG ────────────────────────────────────────────────────
WINDOW      = 24                      # fixed — literature standard
LATENTS     = [2, 4, 6, 8, 12, 16]
HIDDEN_DIM  = 32                      # fixed

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
EPOCHS      = 100
BATCH_SIZE  = 128
LR          = 1e-3
PATIENCE    = 10
THRESH_PCT  = 95
GAP_H       = 6
SMOOTH_WIN  = 4
POINT_K     = 4.0
SMOOTH_K    = 3.0

SEASONS     = {"summer":[6,7,8],"winter":[12,1,2],
               "spring":[3,4,5],"autumn":[9,10,11]}
WEEKEND     = [5, 6]
WORK_START, WORK_END = 8, 18
SPIKE_Q     = 0.9995
COV_THRESH  = 0.35
MIN_READ    = 20
IQR_K       = 3.5
ZERO_H      = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f"{OUT_DIR}/plots", exist_ok=True)

BLUE = "#1565C0"; RED = "#E53935"; GRN = "#2E7D32"
CMAP = plt.cm.tab10(np.linspace(0, 0.85, len(LATENTS)))

# ── MODEL ─────────────────────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.enc1 = nn.LSTM(1,          HIDDEN_DIM, batch_first=True)
        self.enc2 = nn.LSTM(HIDDEN_DIM, latent_dim, batch_first=True)
        self.dec1 = nn.LSTM(latent_dim, HIDDEN_DIM, batch_first=True)
        self.dec2 = nn.LSTM(HIDDEN_DIM, 1,          batch_first=True)

    def forward(self, x):
        o, _ = self.enc1(x); o, _ = self.enc2(o)
        l = o[:, -1:, :].repeat(1, x.size(1), 1)
        o, _ = self.dec1(l); o, _ = self.dec2(o)
        return o

# ── ETL ───────────────────────────────────────────────────────
def get_season(month):
    for name, months in SEASONS.items():
        if month in months: return name
    return "autumn"

def load_and_aggregate(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates().sort_values("date").reset_index(drop=True)
    df = df.drop_duplicates(subset=["sn","date"], keep="first")
    df["month"]  = df["date"].dt.month
    df["season"] = df["month"].apply(get_season)
    global_cap   = df["consumption_diff"].quantile(SPIKE_Q)
    caps = {s: max(df[df["season"]==s]["consumption_diff"].quantile(SPIKE_Q),
                   global_cap*0.8) for s in df["season"].unique()}
    df["cap"] = df["season"].map(caps)
    df.loc[df["consumption_diff"] > df["cap"], "consumption_diff"] = np.nan
    df["dt"]   = df["date"].diff().dt.total_seconds()
    df["hour"] = df["date"].dt.floor("h")
    hourly = (
        df.groupby("hour")
        .agg(sn=("sn","first"), energy_kwh=("consumption_diff","sum"),
             valid_readings=("consumption_diff","count"), median_dt=("dt","median"))
        .reset_index().rename(columns={"hour":"datetime"})
    )
    hourly["median_dt"] = hourly["median_dt"].fillna(22).replace(0,22)
    hourly["expected"]  = (3600 / hourly["median_dt"]).round(2)
    hourly["coverage"]  = hourly["valid_readings"] / hourly["expected"]
    full_idx = pd.date_range(hourly["datetime"].min(),
                             hourly["datetime"].max(), freq="h")
    hourly = hourly.set_index("datetime").reindex(full_idx)
    hourly.index.name = "datetime"
    hourly["sn"] = hourly["sn"].ffill()
    bad = ((hourly["coverage"] < COV_THRESH) |
           (hourly["valid_readings"] < MIN_READ)).fillna(True)
    hourly.loc[bad, "energy_kwh"] = np.nan
    hourly = hourly.reset_index()
    hourly["hour_of_day"] = hourly["datetime"].dt.hour
    hourly["dayofweek"]   = hourly["datetime"].dt.dayofweek
    hourly["month"]       = hourly["datetime"].dt.month
    hourly["season"]      = hourly["month"].apply(get_season)
    hourly["is_weekend"]  = hourly["dayofweek"].isin(WEEKEND).astype(int)
    hourly["is_work"]     = (
        (~hourly["dayofweek"].isin(WEEKEND)) &
        (hourly["hour_of_day"] >= WORK_START) &
        (hourly["hour_of_day"] <= WORK_END)
    ).astype(int)
    return hourly

def clean_for_training(hourly):
    for (season, is_work, is_we, hr), group in hourly.groupby(
            ["season","is_work","is_weekend","hour_of_day"]):
        vals = group["energy_kwh"].dropna()
        if len(vals) < 5: continue
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        bad_idx = group.index[
            group["energy_kwh"].notna() &
            ((group["energy_kwh"] > q3 + IQR_K*iqr) |
             (group["energy_kwh"] < max(0, q1 - IQR_K*iqr)))
        ]
        hourly.loc[bad_idx, "energy_kwh"] = np.nan
    is_zero  = hourly["energy_kwh"] == 0
    zero_grp = (is_zero != is_zero.shift()).cumsum()
    run_len  = is_zero.groupby(zero_grp).transform("sum")
    hourly.loc[is_zero & (run_len > ZERO_H), "energy_kwh"] = np.nan
    hourly["energy_kwh"] = hourly["energy_kwh"].interpolate(method="linear", limit=3)
    rng      = np.random.default_rng(42)
    nan_mask = hourly["energy_kwh"].isna()
    nan_grp  = (nan_mask != nan_mask.shift()).cumsum()
    gap_len  = nan_mask.groupby(nan_grp).transform("sum")
    for idx in hourly[hourly["energy_kwh"].isna()].index:
        row     = hourly.loc[idx]
        season, hr = row["season"], int(row["hour_of_day"])
        is_work = row["is_work"]; dt_val = row["datetime"]
        if gap_len.loc[idx] <= 24:
            w_start = dt_val - pd.Timedelta(days=14)
            w_end   = dt_val + pd.Timedelta(days=14)
            cands = hourly[
                (hourly["datetime"] >= w_start) & (hourly["datetime"] <= w_end) &
                (hourly["hour_of_day"] == hr) & (hourly["season"] == season) &
                (hourly["is_work"] == is_work) & (hourly["energy_kwh"].notna())
            ]["energy_kwh"]
        else:
            cands = hourly[
                (hourly["hour_of_day"] == hr) & (hourly["season"] == season) &
                (hourly["energy_kwh"].notna())
            ]["energy_kwh"]
        if len(cands) < 3:
            cands = hourly[(hourly["hour_of_day"] == hr) &
                           hourly["energy_kwh"].notna()]["energy_kwh"]
        if len(cands) == 0: continue
        mu    = cands.mean()
        sigma = max(cands.std() if len(cands) > 1 else mu*0.15, 0.01)
        hourly.at[idx, "energy_kwh"] = max(0.0, float(rng.normal(mu, sigma*0.8)))
    hourly["energy_kwh"] = gaussian_filter1d(
        hourly["energy_kwh"].fillna(0).clip(lower=0).values, sigma=0.8).clip(0)
    r_mean = hourly["energy_kwh"].rolling(168, center=True, min_periods=24).mean()
    r_std  = hourly["energy_kwh"].rolling(168, center=True, min_periods=24).std()
    r_med  = hourly["energy_kwh"].rolling(168, center=True, min_periods=24).median()
    z      = (hourly["energy_kwh"] - r_mean) / r_std.replace(0, np.nan)
    hourly.loc[z.abs() > 3.5, "energy_kwh"] = r_med[z.abs() > 3.5]
    hourly["energy_kwh"] = hourly["energy_kwh"].clip(lower=0)
    return hourly

# ── HELPERS ───────────────────────────────────────────────────
def make_sequences(values, window):
    X, idx = [], []
    for i in range(len(values) - window + 1):
        X.append(values[i : i + window]); idx.append(i + window - 1)
    return np.array(X, dtype=np.float32)[:, :, np.newaxis], np.array(idx)

def get_errors_recon(model, X):
    loader = DataLoader(TensorDataset(torch.tensor(X)), batch_size=256, shuffle=False)
    errs, recons = [], []
    with torch.no_grad():
        for (b,) in loader:
            b = b.to(DEVICE); r = model(b)
            errs.append(((r - b)**2).mean(dim=[1,2]).cpu().numpy())
            recons.append(r.squeeze(-1).cpu().numpy())
    return np.concatenate(errs), np.concatenate(recons)

def inv(arr, scaler):
    return scaler.inverse_transform(arr.reshape(-1,1)).flatten()

def rolling_threshold(errors, k, window=168, lag=4):
    errors = np.asarray(errors); n = len(errors)
    thresh = np.empty(n)
    g_mean, g_std = errors[:min(window,n)].mean(), errors[:min(window,n)].std()
    thresh[:min(window,n)] = g_mean + k * g_std
    for i in range(min(window,n), n):
        seg = errors[max(0, i-window-lag): max(0, i-lag)]
        thresh[i] = (seg.mean() + k*seg.std()) if len(seg) > 0 else thresh[i-1]
    return thresh

def group_events(ts, mask):
    times = pd.to_datetime(ts[mask])
    if len(times) == 0: return []
    events, s, e = [], times[0], times[0]
    for t in times[1:]:
        if (t - e).total_seconds()/3600 <= GAP_H: e = t
        else:
            events.append((s, e, int((e-s).total_seconds()/3600)+1)); s = e = t
    events.append((s, e, int((e-s).total_seconds()/3600)+1))
    return events

def metrics(actual, predicted):
    mask = actual > 0.01
    return dict(
        mae  = mean_absolute_error(actual, predicted),
        rmse = np.sqrt(mean_squared_error(actual, predicted)),
        r2   = r2_score(actual, predicted),
        mape = np.mean(np.abs((actual[mask]-predicted[mask])/actual[mask]))*100,
    )

# ── TRAIN ONE LATENT ──────────────────────────────────────────
def run_latent(latent_dim, X_train, X_val, X_test,
               ts_test, scaler, n_train, n_val):

    compression = f"{WINDOW}:{latent_dim}"
    ratio       = WINDOW / latent_dim
    if latent_dim < WINDOW:
        ratio_label = f"{ratio:.1f}x compression"
    elif latent_dim == WINDOW:
        ratio_label = "no bottleneck (1:1 pass-through)"
    else:
        ratio_label = f"{latent_dim/WINDOW:.1f}x EXPANSION (no compression)"
    print(f"  latent={latent_dim:2d} | {ratio_label:<32s} | "
          f"train={len(X_train):,} val={len(X_val):,} test={len(X_test):,}")

    model     = LSTMAutoencoder(latent_dim=latent_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=4, min_lr=1e-6)
    loader    = DataLoader(TensorDataset(torch.tensor(X_train)),
                           batch_size=BATCH_SIZE, shuffle=True)
    val_t     = torch.tensor(X_val).to(DEVICE)

    hist = {"train":[], "val":[]}
    best_val, patience_cnt, best_state = float("inf"), 0, None

    for epoch in range(1, EPOCHS+1):
        model.train()
        tl = []
        for (b,) in loader:
            b = b.to(DEVICE); optimizer.zero_grad()
            loss = criterion(model(b), b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); tl.append(loss.item())
        model.eval()
        with torch.no_grad():
            vl = criterion(model(val_t), val_t).item()
        tl = np.mean(tl)
        hist["train"].append(tl); hist["val"].append(vl)
        scheduler.step(vl)
        if vl < best_val:
            best_val, patience_cnt = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE: break

    model.load_state_dict(best_state); model.eval()

    train_errs, _           = get_errors_recon(model, X_train)
    val_errs,   val_recon   = get_errors_recon(model, X_val)
    test_errs,  test_recon  = get_errors_recon(model, X_test)
    threshold = float(np.percentile(train_errs, THRESH_PCT))

    act_v  = inv(X_val[:,-1,0],  scaler); rec_v  = inv(val_recon[:,-1],  scaler)
    act_te = inv(X_test[:,-1,0], scaler); rec_te = inv(test_recon[:,-1], scaler)
    mv  = metrics(act_v,  rec_v)
    mte = metrics(act_te, rec_te)

    roll_thresh   = rolling_threshold(test_errs, k=POINT_K)
    mask_point    = test_errs > roll_thresh
    smooth_errs   = pd.Series(test_errs).rolling(SMOOTH_WIN, min_periods=1).mean().values
    smooth_thresh = rolling_threshold(smooth_errs, k=SMOOTH_K)
    mask_smooth   = smooth_errs > smooth_thresh
    mask          = mask_point | mask_smooth
    events        = group_events(ts_test, mask)

    print(f"         val_R²={mv['r2']:.4f}  val_MAE={mv['mae']:.4f}  "
          f"val_MAPE={mv['mape']:.2f}%  "
          f"test_R²={mte['r2']:.4f}  "
          f"anomaly={mask.mean()*100:.2f}%  events={len(events)}  "
          f"epochs={len(hist['val'])}")

    return dict(
        latent_dim=latent_dim,
        compression=compression,
        compression_ratio=ratio,
        ratio_label=ratio_label,
        best_val_loss=best_val,
        epochs=len(hist["val"]),
        val_r2=mv["r2"],   val_mae=mv["mae"],
        val_rmse=mv["rmse"], val_mape=mv["mape"],
        test_r2=mte["r2"],  test_mae=mte["mae"],
        test_rmse=mte["rmse"], test_mape=mte["mape"],
        threshold=threshold,
        train_err_mean=train_errs.mean(), train_err_std=train_errs.std(),
        val_err_mean=val_errs.mean(),     val_err_std=val_errs.std(),
        test_err_mean=test_errs.mean(),   test_err_std=test_errs.std(),
        anomaly_rate=mask.mean()*100,
        n_flagged=int(mask.sum()), n_events=len(events),
        _hist=hist,
        _ts_test=ts_test, _test_errs=test_errs,
        _act_te=act_te,   _rec_te=rec_te,
        _mask=mask, _events=events,
    )

# ── PLOTS ─────────────────────────────────────────────────────
def plot_r2_curve(results):
    latents  = [r["latent_dim"]  for r in results]
    val_r2s  = [r["val_r2"]      for r in results]
    test_r2s = [r["test_r2"]     for r in results]
    val_maes = [r["val_mae"]     for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Latent Dimension Sweep — Window={WINDOW}h fixed",
                 fontweight="bold", fontsize=14)

    # R² vs latent
    axes[0].plot(latents, val_r2s,  color=BLUE, lw=2.5, marker="o", ms=9,
                 label="Val R² (clean data)")
    axes[0].plot(latents, test_r2s, color=RED,  lw=2.5, marker="s", ms=9,
                 ls="--", label="Test R² (raw data)")
    for l, vr, tr in zip(latents, val_r2s, test_r2s):
        axes[0].annotate(f"{vr:.3f}", (l, vr),
                         textcoords="offset points", xytext=(0, 10),
                         ha="center", fontsize=8, color=BLUE, fontweight="bold")
        axes[0].annotate(f"{tr:.3f}", (l, tr),
                         textcoords="offset points", xytext=(0, -16),
                         ha="center", fontsize=8, color=RED, fontweight="bold")

    best_val_l = latents[int(np.argmax(val_r2s))]
    axes[0].axvline(best_val_l, color=BLUE, ls="--", lw=1.5, alpha=0.6,
                    label=f"Best val: latent={best_val_l}")
    axes[0].set_title("R² vs Latent Dimension", fontweight="bold")
    axes[0].set_xlabel("Latent dimension"); axes[0].set_ylabel("R²")
    axes[0].set_xticks(latents); axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    # MAE vs latent
    axes[1].plot(latents, val_maes, color=GRN, lw=2.5, marker="^", ms=9,
                 label="Val MAE (kWh)")
    for l, m in zip(latents, val_maes):
        axes[1].annotate(f"{m:.4f}", (l, m),
                         textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=8, color=GRN, fontweight="bold")
    best_mae_l = latents[int(np.argmin(val_maes))]
    axes[1].axvline(best_mae_l, color=GRN, ls="--", lw=1.5, alpha=0.6,
                    label=f"Best MAE: latent={best_mae_l}")
    axes[1].set_title("Val MAE vs Latent Dimension", fontweight="bold")
    axes[1].set_xlabel("Latent dimension"); axes[1].set_ylabel("MAE (kWh)")
    axes[1].set_xticks(latents); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/01_r2_and_mae_curve.png", dpi=150)
    plt.close(); print("  v 01_r2_and_mae_curve.png")


def plot_metric_bars(results):
    latents = [str(r["latent_dim"]) for r in results]
    met = [
        ("val_r2",   "Validation R²",       True,  "higher = better"),
        ("val_mae",  "Val MAE (kWh)",        False, "lower = better"),
        ("val_mape", "Val MAPE (%)",         False, "lower = better"),
        ("test_r2",  "Test R² (raw data)",   True,  "higher = better"),
        ("anomaly_rate", "Anomaly Rate (%)", None,  "informational"),
        ("n_events", "Events Detected",      None,  "informational"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Latent Dimension Sweep — Window={WINDOW}h\nAll metrics",
                 fontweight="bold", fontsize=13)

    for ax, (key, title, hb, note) in zip(axes.flatten(), met):
        vals = [r[key] for r in results]
        bars = ax.bar(latents, vals, color=CMAP, alpha=0.85,
                      edgecolor="white", width=0.65)
        if hb is True:
            best_i = int(np.argmax(vals))
        elif hb is False:
            best_i = int(np.argmin(vals))
        else:
            best_i = None

        # Mark the reference (latent=8, our current model)
        ref_i = [r["latent_dim"] for r in results].index(8) \
                if 8 in [r["latent_dim"] for r in results] else None
        if ref_i is not None:
            bars[ref_i].set_edgecolor("black"); bars[ref_i].set_linewidth(2.5)
        if best_i is not None and best_i != ref_i:
            bars[best_i].set_edgecolor("gold"); bars[best_i].set_linewidth(3.0)

        y_rng = (max(vals) - min(vals)) or 0.01
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+y_rng*0.015,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")
        ax.set_title(f"{title}\n({note})", fontweight="bold")
        ax.set_xlabel("Latent dimension"); ax.set_ylabel(title)
        ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/02_metric_bars.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  v 02_metric_bars.png")


def plot_training_curves(results):
    n = len(results); cols = 4; rows = (n+cols-1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows*3.5))
    axes = axes.flatten()
    fig.suptitle(f"Training Curves — Window={WINDOW}h, varying latent",
                 fontweight="bold", fontsize=13)
    for ax, r, color in zip(axes, results, CMAP):
        ep = range(1, len(r["_hist"]["train"])+1)
        ax.plot(ep, r["_hist"]["train"], color=color, lw=2, label="Train")
        ax.plot(ep, r["_hist"]["val"],   color=color, lw=2, ls="--",
                alpha=0.7, label="Val")
        best_ep = int(np.argmin(r["_hist"]["val"]))+1
        ax.axvline(best_ep, color="gray", ls=":", lw=1.2)
        ax.set_title(f"Latent={r['latent_dim']}  ({r['compression']}, "
                     f"{r['ratio_label']})\n"
                     f"best_val={r['best_val_loss']:.5f}  ep={r['epochs']}",
                     fontweight="bold", fontsize=9)
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    for ax in axes[n:]: ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/03_training_curves.png", dpi=150)
    plt.close(); print("  v 03_training_curves.png")


def plot_reconstruction_best(results, scaler):
    """Reconstruction plot for the best latent model on test set."""
    best = max(results, key=lambda r: r["val_r2"])
    n_days = 21; n = n_days * 24
    ts_  = pd.to_datetime(best["_ts_test"][:n])
    act  = best["_act_te"][:n]
    rec  = best["_rec_te"][:n]
    mask = best["_mask"][:n]

    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
    fig.suptitle(
        f"Best Model: Latent={best['latent_dim']} ({best['compression']}, "
        f"{best['ratio_label']})\n"
        f"Val R²={best['val_r2']:.4f}  MAE={best['val_mae']:.4f} kWh  "
        f"MAPE={best['val_mape']:.2f}%  |  "
        f"Test: {best['n_events']} events detected",
        fontweight="bold", fontsize=12)

    axes[0].plot(ts_, act, color=BLUE, lw=1.3, label="Actual (raw)")
    axes[0].plot(ts_, rec, color=RED,  lw=1.3, ls="--", label="Reconstructed")
    axes[0].scatter(ts_[mask], act[mask], color="darkred", s=30,
                    zorder=5, label=f"Anomaly ({mask.sum()} pts)")
    axes[0].set_ylabel("Energy (kWh)"); axes[0].legend(); axes[0].grid(alpha=0.3)

    resid = act - rec
    axes[1].fill_between(ts_, resid, 0, where=resid>=0,
                         color=RED,  alpha=0.4, label="+residual")
    axes[1].fill_between(ts_, resid, 0, where=resid<0,
                         color=BLUE, alpha=0.4, label="-residual")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("Residual (kWh)"); axes[1].set_xlabel("Date")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[1].xaxis.set_major_locator(mdates.DayLocator(interval=3))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/04_best_reconstruction.png", dpi=150)
    plt.close(); print("  v 04_best_reconstruction.png")


def plot_compression_vs_r2(results):
    """Scatter: compression ratio vs Val R² — key insight plot."""
    ratios  = [r["compression_ratio"] for r in results]
    val_r2s = [r["val_r2"]            for r in results]
    latents = [r["latent_dim"]         for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    sc = ax.scatter(ratios, val_r2s, c=latents, cmap="RdYlGn_r",
                    s=120, zorder=5, edgecolors="white", linewidths=1.5)
    for ratio, r2, l in zip(ratios, val_r2s, latents):
        ax.annotate(f"L={l}\n({'compress' if ratio>1 else ('1:1' if ratio==1 else 'expand')})", (ratio, r2),
                    textcoords="offset points", xytext=(8, 0),
                    fontsize=8, fontweight="bold")
    ax.axhline(0, color="red", ls="--", lw=1, alpha=0.5, label="R²=0 (random baseline)")
    plt.colorbar(sc, ax=ax, label="Latent dimension")
    ax.set_title(f"Compression Ratio vs Val R²  (Window={WINDOW}h)\n"
                 "Sweet spot: enough latent dims to represent daily profile",
                 fontweight="bold", fontsize=11)
    ax.set_xlabel("Window / latent  (>1 = compression, 1 = no bottleneck, <1 = expansion)")
    ax.axvline(1.0, color="gray", ls="--", lw=1, alpha=0.6, label="No bottleneck (ratio=1)")
    ax.legend(fontsize=8)
    ax.set_ylabel("Validation R²")
    ax.legend(); ax.grid(alpha=0.3)
    ax.invert_xaxis()   # low ratio (less compression) on right, high ratio on left
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/05_compression_vs_r2.png", dpi=150)
    plt.close(); print("  v 05_compression_vs_r2.png")


def plot_error_distributions(results):
    n = len(results); cols = 4; rows = (n+cols-1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows*3.5))
    axes = axes.flatten()
    fig.suptitle("Test Error Distributions — varying latent",
                 fontweight="bold", fontsize=13)
    for ax, r, color in zip(axes, results, CMAP):
        errs   = r["_test_errs"]
        thresh = r["threshold"]
        ax.hist(errs, bins=50, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(thresh, color="black", ls="--", lw=2,
                   label=f"Threshold={thresh:.5f}")
        ax.set_title(f"Latent={r['latent_dim']}  ({r['compression']})\n"
                     f"Anomaly rate: {r['anomaly_rate']:.2f}%  "
                     f"Events: {r['n_events']}",
                     fontweight="bold", fontsize=9)
        ax.set_xlabel("MSE per 24h window"); ax.set_ylabel("Count")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    for ax in axes[n:]: ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/06_error_distributions.png", dpi=150)
    plt.close(); print("  v 06_error_distributions.png")


# ── REPORT ────────────────────────────────────────────────────
def print_report(results):
    print("\n" + "="*82)
    print(f"  LATENT DIMENSION SWEEP REPORT  (Window={WINDOW}h fixed)")
    print("="*82)
    print(f"\n  {'Latent':>8}  {'Ratio':>8}  {'Val R²':>7}  {'Val MAE':>8}  "
          f"{'Val MAPE':>9}  {'Test R²':>7}  {'Anom%':>6}  "
          f"{'Events':>7}  {'Epochs':>7}")
    print("  " + "-"*82)

    ref_r2 = next(r["val_r2"] for r in results if r["latent_dim"] == 8) \
             if any(r["latent_dim"]==8 for r in results) else results[0]["val_r2"]

    for r in results:
        mark = "  <- current model" if r["latent_dim"] == 8 else ""
        print(f"  {r['latent_dim']:>8}  {r['ratio_label']:>28s}"
              f"  {r['val_r2']:>7.4f}  {r['val_mae']:>8.4f}  "
              f"{r['val_mape']:>8.2f}%  {r['test_r2']:>7.4f}  "
              f"{r['anomaly_rate']:>5.2f}%  {r['n_events']:>7}  "
              f"{r['epochs']:>7}{mark}")

    best_val  = max(results, key=lambda r: r["val_r2"])
    best_test = max(results, key=lambda r: r["test_r2"])
    best_mae  = min(results, key=lambda r: r["val_mae"])

    print(f"\n  Best Val R²  : latent={best_val['latent_dim']}"
          f"  R²={best_val['val_r2']:.4f}")
    print(f"  Best Test R² : latent={best_test['latent_dim']}"
          f"  R²={best_test['test_r2']:.4f}")
    print(f"  Best Val MAE : latent={best_mae['latent_dim']}"
          f"  MAE={best_mae['val_mae']:.4f} kWh")

    print("\n" + "="*82)
    print("  RECOMMENDATION")
    print("="*82)

    # Sweet spot: best val R² while maintaining meaningful detection
    scored = [(r["val_r2"] + r["test_r2"]) / 2 for r in results]
    best_overall_i = int(np.argmax(scored))
    best_overall   = results[best_overall_i]

    print(f"\n  Optimal latent dimension : {best_overall['latent_dim']}")
    print(f"  Compression ratio        : {best_overall['ratio_label']}")
    print(f"  Val R²                   : {best_overall['val_r2']:.4f}")
    print(f"  Val MAE                  : {best_overall['val_mae']:.4f} kWh")
    print(f"  Test R²                  : {best_overall['test_r2']:.4f}")
    print(f"  Events detected          : {best_overall['n_events']}")

    print(f"\n  Interpretation:")
    print(f"    Latent too small relative to window (over-compression)")
    print(f"    -> model can't represent diversity of daily profiles -> poor R2")
    print(f"    Latent too close to or exceeding window length (no real bottleneck)")
    print(f"    → bottleneck too loose → model memorises anomalies → poor detection")
    print(f"    Sweet spot at latent={best_overall['latent_dim']} "
          f"({best_overall['ratio_label']})")
    print("="*82)


# ── MAIN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nDevice  : {DEVICE}")
    print(f"Window  : {WINDOW}h (fixed)")
    print(f"Latents : {LATENTS}")
    print(f"Hidden  : {HIDDEN_DIM}  |  Epochs: {EPOCHS}  |  "
          f"Patience: {PATIENCE}  |  Batch: {BATCH_SIZE}")
    print(f"Ratios  : {[f'{WINDOW}:{l} ({WINDOW/l:.1f}:1)' for l in LATENTS]}\n")

    # ── ETL (run once, reuse for all latent experiments) ─────
    print("Running ETL from raw_data.csv ...")
    print("  [shared] Load & aggregate...")
    df_base = load_and_aggregate(RAW_CSV)

    print("  [TRAIN path] Full cleaning (IQR + impute + smooth)...")
    df_clean = clean_for_training(df_base.copy())
    df_clean = df_clean.sort_values("datetime").reset_index(drop=True)

    print("  [TEST path]  Minimal (ffill only, anomalies kept)...")
    df_raw   = df_base.copy()
    df_raw["energy_kwh"] = df_raw["energy_kwh"].ffill().bfill().clip(lower=0)
    df_raw   = df_raw.sort_values("datetime").reset_index(drop=True)

    n_total = len(df_clean)
    n_train = int(n_total * TRAIN_RATIO)
    n_val   = int(n_total * VAL_RATIO)

    energy_clean = df_clean["energy_kwh"].values.astype(np.float32)
    energy_raw   = df_raw["energy_kwh"].values.astype(np.float32)

    scaler = MinMaxScaler()
    scaler.fit(energy_clean[:n_train].reshape(-1,1))

    print(f"  Clean rows : {n_total:,}  "
          f"({df_clean['datetime'].min().date()} → "
          f"{df_clean['datetime'].max().date()})")
    print(f"  Clean range: {energy_clean.min():.3f} – {energy_clean.max():.3f} kWh")
    print(f"  Raw range  : {energy_raw.min():.3f} – {energy_raw.max():.3f} kWh\n")

    # ── Build sequences once (same for all latent experiments) ─
    scaled_clean = scaler.transform(energy_clean.reshape(-1,1)).flatten()
    scaled_raw   = scaler.transform(energy_raw.reshape(-1,1)).flatten()

    X_all_c, idx_c = make_sequences(scaled_clean, WINDOW)
    X_all_r, idx_r = make_sequences(scaled_raw,   WINDOW)

    ts_c = df_clean["datetime"].values[idx_c]
    ts_r = df_raw["datetime"].values[idx_r]

    n_seq_tr = n_train - WINDOW + 1
    n_seq_v  = n_val

    X_train  = X_all_c[:n_seq_tr]
    X_val    = X_all_c[n_seq_tr : n_seq_tr + n_seq_v]
    X_test   = X_all_r[n_seq_tr + n_seq_v:]
    ts_test  = ts_r[n_seq_tr + n_seq_v:]

    print(f"  Sequences — Train: {len(X_train):,}  "
          f"Val: {len(X_val):,}  Test: {len(X_test):,}\n")

    # ── Run all latent experiments ────────────────────────────
    results = []
    for i, l in enumerate(LATENTS, 1):
        print(f"[{i}/{len(LATENTS)}] Latent = {l}", flush=True)
        r = run_latent(l, X_train, X_val, X_test,
                       ts_test, scaler, n_train, n_val)
        results.append(r)

    # Save CSV
    df_out = pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in results
    ])
    df_out.to_csv(f"{OUT_DIR}/results.csv", index=False)
    print(f"\nSaved -> {OUT_DIR}/results.csv")

    # Plots
    print("\nGenerating plots...")
    plot_r2_curve(results)
    plot_metric_bars(results)
    plot_training_curves(results)
    plot_reconstruction_best(results, scaler)
    plot_compression_vs_r2(results)
    plot_error_distributions(results)

    print_report(results)
    print(f"\nAll outputs -> {OUT_DIR}/")