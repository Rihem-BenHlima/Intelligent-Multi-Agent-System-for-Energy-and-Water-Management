"""
latent_sweep_water.py — Latent Dimension Sweep (Window=24h fixed, WATER)
Tests latent dims [2,4,6,8,12,16]. Same ETL as step1_etl_water.py (water-adapted:
hourly, no zero-streak rule, no smoothing, IQR on non-zeros, hour_of_day context only).
Fixed threshold (mean+k·std of train errors). Two-path ETL: train on clean, test on raw.
Outputs: latent_sweep_water/results.csv, latent_sweep_water/plots/
Run: python latent_sweep_water.py
"""

import os, sys, warnings, pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import WATER_CSV, LATENT_SWEEP_DIR as OUT_DIR

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
IQR_K       = 3.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f"{OUT_DIR}/plots", exist_ok=True)

BLUE = "#1565C0"; RED = "#E53935"; GRN = "#2E7D32"
CMAP = plt.cm.tab10(np.linspace(0, 0.85, len(LATENTS)))

# ── MODEL (unchanged) ─────────────────────────────────────────
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

# ── ETL (WATER version, NO season/weekend context — see step1_etl_water.py) ────
def load_consumption_liters(csv_path):
    """Read water CSV; auto-converts consumption_m3→consumption_l (×1000) if needed."""
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    if "consumption_l" in df.columns:
        df["consumption_l"] = df["consumption_l"].astype(float)
    elif "consumption_m3" in df.columns:
        df["consumption_l"] = df["consumption_m3"].astype(float) * 1000.0
    else:
        raise KeyError(
            f"{csv_path} has neither 'consumption_l' nor 'consumption_m3' "
            f"column — found: {list(df.columns)}"
        )
    return df


def load_hourly(path):
    """Load hourly water data with hour_of_day context. No season/weekend split."""
    df = load_consumption_liters(path).sort_values("datetime")
    df = df.drop_duplicates(subset="datetime").reset_index(drop=True)

    hourly = df[["datetime"]].copy()
    hourly["consumption_l"] = df["consumption_l"]   # L/hour
    hourly["sn"] = "Panther_office_Larry"

    full_idx = pd.date_range(hourly["datetime"].min(),
                             hourly["datetime"].max(), freq="h")
    hourly = (hourly.set_index("datetime").reindex(full_idx)
                    .rename_axis("datetime").reset_index())
    hourly["sn"] = hourly["sn"].ffill()

    hourly["hour_of_day"] = hourly["datetime"].dt.hour
    hourly["dayofweek"]   = hourly["datetime"].dt.dayofweek   # metadata only

    global_cap = hourly["consumption_l"].quantile(SPIKE_Q)
    spikes = hourly["consumption_l"] > global_cap
    hourly.loc[spikes, "consumption_l"] = np.nan
    return hourly

def clean_for_training(hourly):
    """IQR on non-zero values (by hour_of_day) + zero-inflated imputation. No smoothing, no zero-streak rule."""
    for hr, group in hourly.groupby("hour_of_day"):
        vals = group.loc[group["consumption_l"] > 0, "consumption_l"].dropna()
        if len(vals) < 5: continue
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        hi  = q3 + IQR_K * iqr
        bad_idx = group.index[group["consumption_l"].notna() &
                              (group["consumption_l"] > hi)]
        hourly.loc[bad_idx, "consumption_l"] = np.nan

    hourly["consumption_l"] = hourly["consumption_l"].interpolate(method="linear", limit=3)

    rng      = np.random.default_rng(42)
    nan_mask = hourly["consumption_l"].isna()
    nan_grp  = (nan_mask != nan_mask.shift()).cumsum()
    gap_len  = nan_mask.groupby(nan_grp).transform("sum")

    for idx in hourly[hourly["consumption_l"].isna()].index:
        row = hourly.loc[idx]
        hr, dt_val = int(row["hour_of_day"]), row["datetime"]

        if gap_len.loc[idx] <= 24:
            w_s, w_e = dt_val - pd.Timedelta(days=14), dt_val + pd.Timedelta(days=14)
            cands = hourly[
                (hourly["datetime"].between(w_s, w_e)) &
                (hourly["hour_of_day"] == hr) &
                hourly["consumption_l"].notna()]["consumption_l"]
        else:
            cands = hourly[
                (hourly["hour_of_day"] == hr) &
                hourly["consumption_l"].notna()]["consumption_l"]

        if len(cands) < 3:
            cands = hourly[hourly["consumption_l"].notna()]["consumption_l"]
        if len(cands) == 0: continue

        p_zero = (cands == 0).mean()
        if rng.random() < p_zero:
            hourly.at[idx, "consumption_l"] = 0.0
        else:
            nz = cands[cands > 0]
            mu = nz.mean()
            sigma = max(nz.std() if len(nz) > 1 else mu * 0.15, 0.01)
            hourly.at[idx, "consumption_l"] = max(0.0, float(rng.normal(mu, sigma * 0.8)))

    e = hourly["consumption_l"].clip(lower=0)
    r_mean = e.rolling(168, center=True, min_periods=24).mean()
    r_std  = e.rolling(168, center=True, min_periods=24).std()
    r_med  = e.rolling(168, center=True, min_periods=24).median()
    z = (e - r_mean) / r_std.replace(0, np.nan)
    e[z.abs() > 3.5] = r_med[z.abs() > 3.5]
    hourly["consumption_l"] = e.clip(lower=0)
    return hourly

def prepare_for_detection(hourly, orig_path):
    """RAW path — restore capped spikes, ffill true gaps only."""
    orig = load_consumption_liters(orig_path)
    orig = orig.drop_duplicates(subset="datetime").set_index("datetime")["consumption_l"]
    restored = hourly.set_index("datetime")["consumption_l"].copy()
    mask = restored.isna() & orig.reindex(restored.index).notna()
    restored[mask] = orig.reindex(restored.index)[mask]
    hourly["consumption_l"] = restored.values
    hourly["consumption_l"] = hourly["consumption_l"].ffill().bfill().clip(lower=0)
    return hourly

# ── HELPERS (unchanged) ────────────────────────────────────────
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

def fixed_threshold(baseline_errors, k):
    """Fixed threshold: mean + k·std. Applied flat — no rolling window."""
    baseline_errors = np.asarray(baseline_errors, dtype=np.float64)
    return float(baseline_errors.mean() + k * baseline_errors.std())

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
    mask = actual > 10  # floor: ignore near-zero actuals in MAPE
    return dict(
        mae  = mean_absolute_error(actual, predicted),
        rmse = np.sqrt(mean_squared_error(actual, predicted)),
        r2   = r2_score(actual, predicted),
        mape = np.mean(np.abs((actual[mask]-predicted[mask])/actual[mask]))*100
               if mask.sum() > 0 else np.nan,
    )

# Train one latent configuration
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

    train_smooth  = pd.Series(train_errs).rolling(SMOOTH_WIN, min_periods=1).mean().values
    point_thresh  = fixed_threshold(train_errs, k=POINT_K)
    mask_point    = test_errs > point_thresh
    smooth_errs   = pd.Series(test_errs).rolling(SMOOTH_WIN, min_periods=1).mean().values
    smooth_thresh = fixed_threshold(train_smooth, k=SMOOTH_K)
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

def plot_r2_curve(results):
    latents  = [r["latent_dim"]  for r in results]
    val_r2s  = [r["val_r2"]      for r in results]
    test_r2s = [r["test_r2"]     for r in results]
    val_maes = [r["val_mae"]     for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Latent Dimension Sweep (WATER) — Window={WINDOW}h fixed",
                 fontweight="bold", fontsize=14)

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

    axes[1].plot(latents, val_maes, color=GRN, lw=2.5, marker="^", ms=9,
                 label="Val MAE (L/h)")
    for l, m in zip(latents, val_maes):
        axes[1].annotate(f"{m:.2f}", (l, m),
                         textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=8, color=GRN, fontweight="bold")
    best_mae_l = latents[int(np.argmin(val_maes))]
    axes[1].axvline(best_mae_l, color=GRN, ls="--", lw=1.5, alpha=0.6,
                    label=f"Best MAE: latent={best_mae_l}")
    axes[1].set_title("Val MAE vs Latent Dimension", fontweight="bold")
    axes[1].set_xlabel("Latent dimension"); axes[1].set_ylabel("MAE (L/h)")
    axes[1].set_xticks(latents); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/01_r2_and_mae_curve.png", dpi=150)
    plt.close(); print("  v 01_r2_and_mae_curve.png")


def plot_metric_bars(results):
    latents = [str(r["latent_dim"]) for r in results]
    met = [
        ("val_r2",   "Validation R²",       True,  "higher = better"),
        ("val_mae",  "Val MAE (L/h)",        False, "lower = better"),
        ("val_mape", "Val MAPE (%)",         False, "lower = better"),
        ("test_r2",  "Test R² (raw data)",   True,  "higher = better"),
        ("anomaly_rate", "Anomaly Rate (%)", None,  "informational"),
        ("n_events", "Events Detected",      None,  "informational"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Latent Dimension Sweep (WATER) — Window={WINDOW}h\nAll metrics",
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
    fig.suptitle(f"Training Curves (WATER) — Window={WINDOW}h, varying latent",
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
    best = max(results, key=lambda r: r["val_r2"])
    n_days = 21; n = n_days * 24
    ts_  = pd.to_datetime(best["_ts_test"][:n])
    act  = best["_act_te"][:n]
    rec  = best["_rec_te"][:n]
    mask = best["_mask"][:n]

    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
    fig.suptitle(
        f"Best Model (WATER): Latent={best['latent_dim']} ({best['compression']}, "
        f"{best['ratio_label']})\n"
        f"Val R²={best['val_r2']:.4f}  MAE={best['val_mae']:.2f} L/h  "
        f"MAPE={best['val_mape']:.2f}%  |  "
        f"Test: {best['n_events']} events detected",
        fontweight="bold", fontsize=12)

    axes[0].plot(ts_, act, color=BLUE, lw=1.3, label="Actual (raw)")
    axes[0].plot(ts_, rec, color=RED,  lw=1.3, ls="--", label="Reconstructed")
    axes[0].scatter(ts_[mask], act[mask], color="darkred", s=30,
                    zorder=5, label=f"Anomaly ({mask.sum()} pts)")
    axes[0].set_ylabel("Consumption (L/h)"); axes[0].legend(); axes[0].grid(alpha=0.3)

    resid = act - rec
    axes[1].fill_between(ts_, resid, 0, where=resid>=0,
                         color=RED,  alpha=0.4, label="+residual")
    axes[1].fill_between(ts_, resid, 0, where=resid<0,
                         color=BLUE, alpha=0.4, label="-residual")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("Residual (L/h)"); axes[1].set_xlabel("Date")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[1].xaxis.set_major_locator(mdates.DayLocator(interval=3))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/04_best_reconstruction.png", dpi=150)
    plt.close(); print("  v 04_best_reconstruction.png")


def plot_compression_vs_r2(results):
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
    ax.set_title(f"Compression Ratio vs Val R²  (WATER, Window={WINDOW}h)\n"
                 "Sweet spot: enough latent dims to represent daily on/off profile",
                 fontweight="bold", fontsize=11)
    ax.set_xlabel("Window / latent  (>1 = compression, 1 = no bottleneck, <1 = expansion)")
    ax.axvline(1.0, color="gray", ls="--", lw=1, alpha=0.6, label="No bottleneck (ratio=1)")
    ax.legend(fontsize=8)
    ax.set_ylabel("Validation R²")
    ax.legend(); ax.grid(alpha=0.3)
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/05_compression_vs_r2.png", dpi=150)
    plt.close(); print("  v 05_compression_vs_r2.png")


def plot_error_distributions(results):
    n = len(results); cols = 4; rows = (n+cols-1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows*3.5))
    axes = axes.flatten()
    fig.suptitle("Test Error Distributions (WATER) — varying latent",
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


def print_report(results):
    print("\n" + "="*82)
    print(f"  LATENT DIMENSION SWEEP REPORT — WATER (Window={WINDOW}h fixed)")
    print("="*82)
    print(f"\n  {'Latent':>8}  {'Ratio':>8}  {'Val R²':>7}  {'Val MAE':>8}  "
          f"{'Val MAPE':>9}  {'Test R²':>7}  {'Anom%':>6}  "
          f"{'Events':>7}  {'Epochs':>7}")
    print("  " + "-"*82)

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
          f"  MAE={best_mae['val_mae']:.2f} L/h")

    print("\n" + "="*82)
    print("  RECOMMENDATION")
    print("="*82)

    scored = [(r["val_r2"] + r["test_r2"]) / 2 for r in results]
    best_overall_i = int(np.argmax(scored))
    best_overall   = results[best_overall_i]

    print(f"\n  Optimal latent dimension : {best_overall['latent_dim']}")
    print(f"  Compression ratio        : {best_overall['ratio_label']}")
    print(f"  Val R²                   : {best_overall['val_r2']:.4f}")
    print(f"  Val MAE                  : {best_overall['val_mae']:.2f} L/h")
    print(f"  Test R²                  : {best_overall['test_r2']:.4f}")
    print(f"  Events detected          : {best_overall['n_events']}")
    print("="*82)


# ── MAIN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nDevice  : {DEVICE}")
    print(f"Window  : {WINDOW}h (fixed)")
    print(f"Latents : {LATENTS}")
    print(f"Hidden  : {HIDDEN_DIM}  |  Epochs: {EPOCHS}  |  "
          f"Patience: {PATIENCE}  |  Batch: {BATCH_SIZE}")
    print(f"Ratios  : {[f'{WINDOW}:{l} ({WINDOW/l:.1f}:1)' for l in LATENTS]}\n")

    print(f"Running ETL from {WATER_CSV} ...")
    print("  [shared] Load hourly + context columns...")
    df_base = load_hourly(WATER_CSV)

    print("  [TRAIN path] IQR (non-zero) + zero-inflated impute (no smoothing)...")
    df_clean = clean_for_training(df_base.copy())
    df_clean = df_clean.sort_values("datetime").reset_index(drop=True)

    print("  [TEST path]  Restore spikes + ffill true gaps only (anomalies kept)...")
    df_raw   = prepare_for_detection(df_base.copy(), WATER_CSV)
    df_raw   = df_raw.sort_values("datetime").reset_index(drop=True)

    n_total = len(df_clean)
    n_train = int(n_total * TRAIN_RATIO)
    n_val   = int(n_total * VAL_RATIO)

    water_clean = df_clean["consumption_l"].values.astype(np.float32)
    water_raw   = df_raw["consumption_l"].values.astype(np.float32)

    scaler = MinMaxScaler()
    scaler.fit(water_clean[:n_train].reshape(-1,1))

    print(f"  Clean rows : {n_total:,}  "
          f"({df_clean['datetime'].min().date()} → "
          f"{df_clean['datetime'].max().date()})")
    print(f"  Clean range: {water_clean.min():.2f} – {water_clean.max():.2f} L/h")
    print(f"  Raw range  : {water_raw.min():.2f} – {water_raw.max():.2f} L/h\n")

    scaled_clean = scaler.transform(water_clean.reshape(-1,1)).flatten()
    scaled_raw   = scaler.transform(water_raw.reshape(-1,1)).flatten()

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

    results = []
    for i, l in enumerate(LATENTS, 1):
        print(f"[{i}/{len(LATENTS)}] Latent = {l}", flush=True)
        r = run_latent(l, X_train, X_val, X_test,
                       ts_test, scaler, n_train, n_val)
        results.append(r)

    df_out = pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in results
    ])
    df_out.to_csv(f"{OUT_DIR}/results.csv", index=False)
    print(f"\nSaved -> {OUT_DIR}/results.csv")

    print("\nGenerating plots...")
    plot_r2_curve(results)
    plot_metric_bars(results)
    plot_training_curves(results)
    plot_reconstruction_best(results, scaler)
    plot_compression_vs_r2(results)
    plot_error_distributions(results)

    print_report(results)
    print(f"\nAll outputs -> {OUT_DIR}/")