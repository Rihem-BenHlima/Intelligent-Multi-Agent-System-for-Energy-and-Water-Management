"""
step3_detect.py — Anomaly Detection
=====================================
Input : etl_output/etl_raw.csv   (from step1_etl.py — anomalies preserved)
        model/        (from step2_train.py)
Output: results/anomaly_results.csv, results/anomaly_events.csv, plots/detect/

Detection methodology:
  - Reconstruction error (MSE) on raw test data (anomalies preserved).
  - Dual-signal detection:
      Signal 1 (point) : MSE > adaptive rolling threshold
                         thresh[i] = mean(e[i-W-lag:i-lag]) + k·std(...)
                         Window W=168h, lag=4h, k=POINT_K
                         Errors clipped at p95 before entering baseline
                         (anti-inflation). Floor = FLOOR_PCT-th pct of val errors.
      Signal 2 (smooth): same formula on 4h rolling mean of MSE, k=SMOOTH_K.
                         Catches sustained drifts missed by point signal.
  - Combined: OR of both signals.
  - Consecutive flagged windows within 6h merged into anomaly events.

Run:  python step3_detect.py
"""

import sys, os, warnings, pickle
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

# Set before torch inits its CUDA/cuBLAS context so
# torch.use_deterministic_algorithms() can make GEMMs reproducible.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import (RAW_CSV, MODEL_DIR, PLOTS_DETECT_DIR as PLOT_DIR,
                     RESULTS_DIR as RESULT_DIR)

# ── CONFIG ────────────────────────────────────────────────────
WINDOW      = 24
# TRAIN_RATIO/VAL_RATIO used only to locate the test boundary (pool_end),
# matching step2_train.py exactly. Val set loaded from val_mask.npy.
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
HIDDEN_DIM  = 32
LATENT_DIM  = 12
GAP_H       = 6
MIN_EVENT_H = 2     # drop single-hour noise flags
SMOOTH_WIN  = 4

# Display-only: clip plot x-axis at this timestamp. Set to None for full range.
PLOT_X_MAX  = "2026-08-27 23:00"

THRESH_WIN  = 168   # rolling window width (hours)
THRESH_LAG  = 4     # lag: baseline ends this many hours before current point
# Tuned via tune_thresholds.py elbow method. Re-run after model/ETL/FLOOR_PCT changes.
POINT_K     = 3.5
SMOOTH_K    = 6.0
LOW_CONF_MARGIN = 2.0  # events with margin < this are flagged for manual review
FLOOR_PCT   = 80    # percentile of RAW val errors for both signal floors.
                    # p80 lets the rolling mean+k*std drive the threshold in
                    # calm periods; it clamps only at the genuine noise floor.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
for d in [PLOT_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

BLUE = "#1565C0"; RED = "#E53935"; GRN = "#2E7D32"; ORG = "#E65100"

# ── MODEL ─────────────────────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1   = nn.LSTM(1,          HIDDEN_DIM, batch_first=True)
        self.enc2   = nn.LSTM(HIDDEN_DIM, LATENT_DIM, batch_first=True)
        self.dec1   = nn.LSTM(LATENT_DIM, HIDDEN_DIM, batch_first=True)
        self.output = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x):
        o, _ = self.enc1(x)
        o, _ = self.enc2(o)
        l = o[:, -1:, :].repeat(1, x.size(1), 1)
        o, _ = self.dec1(l)
        return self.output(o)

# ── HELPERS ───────────────────────────────────────────────────
def _empty_seqs(window):
    return (np.empty((0, window, 1), dtype=np.float32), np.empty(0, dtype=int))

def make_sequences(values, window):
    X, idx = [], []
    for i in range(len(values) - window + 1):
        X.append(values[i : i + window]); idx.append(i + window - 1)
    if not X:
        return _empty_seqs(window)
    return np.array(X, dtype=np.float32)[:, :, np.newaxis], np.array(idx)

def make_sequences_from_mask(values, valid_mask, window):
    """Only sequences fully contained within a contiguous valid_mask
    region - mirrors step2_train.py's helper of the same purpose, so test
    (built from the reserved tail) and val (loaded from val_mask.npy)
    never include a sequence whose window straddles a mask boundary."""
    X, idx = [], []
    for i in range(len(values) - window + 1):
        if valid_mask[i:i + window].all():
            X.append(values[i:i + window]); idx.append(i + window - 1)
    if not X:   # no full window fits — return correctly-shaped empties
        return _empty_seqs(window)
    return np.array(X, dtype=np.float32)[:, :, np.newaxis], np.array(idx)

def get_errors_recon(model, X):
    loader = DataLoader(TensorDataset(torch.tensor(X)), batch_size=512, shuffle=False)
    errs, recons = [], []
    with torch.no_grad():
        for (b,) in loader:
            b = b.to(DEVICE); r = model(b)
            errs.append(((r - b)**2).mean(dim=[1,2]).cpu().numpy())
            recons.append(r.squeeze(-1).cpu().numpy())
    return np.concatenate(errs), np.concatenate(recons)

def inv(arr, scaler):
    return scaler.inverse_transform(arr.reshape(-1,1)).flatten()

def rolling_threshold(errors, k, window=168, lag=4, floor=None, seed=None):
    """
    Adaptive rolling threshold: mean + k·std on a sliding window of past
    errors with a lag separating baseline from the current point.
    Anti-inflation: errors clipped at p95 before entering the rolling baseline.
    Floor: threshold never drops below this value.
    Seed: optional array prepended before computing rolling stats, then stripped.
    """
    errors = np.asarray(errors, dtype=np.float64)

    # Prepend seed if provided
    if seed is not None:
        seed   = np.asarray(seed, dtype=np.float64)
        n_seed = len(seed)
        errors = np.concatenate([seed, errors])
    else:
        n_seed = 0

    clip_val = np.percentile(errors[n_seed:], 95)
    eb       = np.clip(errors, 0, clip_val)

    s      = pd.Series(eb)
    rm     = s.rolling(window, min_periods=1).mean()
    rs     = s.rolling(window, min_periods=1).std().fillna(0)
    thresh = (rm.shift(lag).fillna(rm.iloc[0]) +
              k * rs.shift(lag).fillna(rs.iloc[0])).values

    # Strip the seed portion back off — output length matches input test errors
    thresh = thresh[n_seed:]

    if floor is not None:
        thresh = np.maximum(thresh, floor)
    return thresh

def group_events(ts, mask, gap_h=GAP_H):
    """
    Merge consecutive flagged timestamps into events.

    gap_h is measured between consecutive *flagged* timestamps, not between
    event boundaries. Two flags separated by a clean gap of ≤ gap_h hours
    (with no flags in between) are merged into a single event. Duration is
    wall-clock distance from first to last flagged timestamp + 1h.
    """
    times = pd.to_datetime(ts[mask])
    if len(times) == 0: return []
    events, s, e = [], times[0], times[0]
    for t in times[1:]:
        if (t - e).total_seconds()/3600 <= gap_h: e = t
        else:
            events.append((s, e, int((e-s).total_seconds()/3600)+1)); s = e = t
    events.append((s, e, int((e-s).total_seconds()/3600)+1))
    return events

def compute_metrics(actual, predicted):
    mask = actual > 0.01
    return dict(
        mae  = mean_absolute_error(actual, predicted),
        rmse = np.sqrt(mean_squared_error(actual, predicted)),
        r2   = r2_score(actual, predicted),
        mape = (np.mean(np.abs((actual[mask]-predicted[mask])/actual[mask]))*100
                if mask.any() else float("nan")),
    )

# ── PLOTS ─────────────────────────────────────────────────────
def plot_error_timeline(ts, errors, smooth_err,
                        roll_thresh, smooth_thresh,
                        mask_pt, mask_sm, mask, threshold, smooth_floor):
    ts_ = pd.to_datetime(ts)
    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
    fig.suptitle(
        f"Dual-Signal Anomaly Detection — Test Set (raw data, anomalies preserved)\n"
        f"Combined (OR): {mask.sum()} flagged  |  "
        f"Point: {mask_pt.sum()}  |  Smooth: {mask_sm.sum()}",
        fontweight="bold", y=1.01, fontsize=13)

    axes[0].plot(ts_, errors, color=BLUE, lw=0.7, alpha=0.85,
                 label="Reconstruction error (MSE)")
    axes[0].plot(ts_, roll_thresh, color=RED, lw=1.5, ls="--",
                 label=f"Adaptive threshold")
    axes[0].scatter(ts_[mask_pt], errors[mask_pt], color="darkred",
                    s=20, zorder=5, label=f"Flagged ({mask_pt.sum()})")
    axes[0].set_title(
        f"Point Anomaly Detection — Reconstruction Error vs Adaptive Threshold (k={POINT_K})",
        fontweight="bold")
    axes[0].set_ylabel("MSE"); axes[0].legend(fontsize=9); axes[0].grid(alpha=0.25)

    axes[1].plot(ts_, smooth_err, color=GRN, lw=0.9,
                 label=f"Smoothed error ({SMOOTH_WIN}h rolling mean)")
    axes[1].plot(ts_, smooth_thresh, color=ORG, lw=1.5, ls="--",
                 label=f"Adaptive threshold")
    axes[1].scatter(ts_[mask_sm], smooth_err[mask_sm], color="saddlebrown",
                    s=20, zorder=5, label=f"Flagged ({mask_sm.sum()})")
    axes[1].set_title(
        f"Sustained Drift Detection — Smoothed Reconstruction Error vs Adaptive Threshold (k={SMOOTH_K})",
        fontweight="bold")
    axes[1].set_ylabel("Smoothed MSE"); axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.25)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha="right")
    if PLOT_X_MAX is not None:   # display-only x-axis clip (see CONFIG)
        axes[1].set_xlim(right=pd.Timestamp(PLOT_X_MAX))
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/01_error_timeline.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  ✓ 01_error_timeline.png")

def plot_event_calendar(events, ts_range):
    fig, ax = plt.subplots(figsize=(16, 3.5))
    if not events:
        ax.text(0.5, 0.5, "No anomaly events detected",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
    else:
        cmap = plt.cm.Reds(np.linspace(0.4, 0.9, len(events)))
        for (start, end, dur), color in zip(sorted(events), cmap):
            width = max((end-start).total_seconds()/86400 + 1/24, 1/24)
            ax.barh(0.5, width, left=mdates.date2num(start),
                    height=0.35, color=color, edgecolor="white", alpha=0.9)
            if dur >= 3:
                ax.text(mdates.date2num(start)+width/2, 0.5,
                        f"{dur}h", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        x_right = pd.to_datetime(ts_range[-1])
        if PLOT_X_MAX is not None:   # display-only x-axis clip (see CONFIG)
            x_right = min(x_right, pd.Timestamp(PLOT_X_MAX))
        ax.set_xlim(mdates.date2num(pd.to_datetime(ts_range[0])),
                    mdates.date2num(x_right))
        ax.set_yticks([0.5]); ax.set_yticklabels(["Anomaly events"])
    ax.set_title(f"Anomaly Event Calendar — {len(events)} events detected",
                 fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/02_event_calendar.png", dpi=150)
    plt.close(); print("  ✓ 02_event_calendar.png")

def plot_test_reconstruction(ts, actual, recon, mask, n_days=21):
    n   = min(n_days*24, len(ts))
    ts_ = pd.to_datetime(ts[:n])
    act = actual[:n]; rec = recon[:n]; m = mask[:n]
    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
    fig.suptitle(
        f"Test Set — Actual vs Reconstructed (first {n_days} days)\n"
        f"[Raw data — real anomalies preserved for detection]",
        fontweight="bold", fontsize=13)
    ORANGE = "#FF6D00"
    axes[0].plot(ts_, act, color=BLUE,   lw=1.3, label="Actual (raw)")
    axes[0].plot(ts_, rec, color=ORANGE, lw=1.3, ls="--",
                 label="Reconstructed (expected normal)")
    axes[0].scatter(ts_[m], act[m], color="darkred", s=25, zorder=5,
                    label=f"Anomaly ({m.sum()} pts)")
    axes[0].set_ylabel("Energy (kWh)"); axes[0].legend(); axes[0].grid(alpha=0.3)
    resid = act - rec
    axes[1].fill_between(ts_, resid, 0, where=resid>=0,
                         color=RED,  alpha=0.4, label="+residual (over-consumption)")
    axes[1].fill_between(ts_, resid, 0, where=resid<0,
                         color=BLUE, alpha=0.4, label="-residual (under-consumption)")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("Residual (kWh)"); axes[1].set_xlabel("Date")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[1].xaxis.set_major_locator(mdates.DayLocator(interval=3))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/03_test_reconstruction.png", dpi=150)
    plt.close(); print("  ✓ 03_test_reconstruction.png")

def plot_season_breakdown(df, ts, errors, mask):
    ts_ = pd.to_datetime(ts)
    df_e = pd.DataFrame({"datetime": ts_, "error": errors, "anomaly": mask})
    df_e = df_e.merge(df[["datetime","season"]].drop_duplicates("datetime"),
                      on="datetime", how="left")
    seasons = ["spring","summer","autumn","winter"]
    colors  = [GRN, RED, ORG, BLUE]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Per-Season Analysis (test set — raw data)",
                 fontweight="bold", fontsize=13)
    rates  = [df_e[df_e["season"]==s]["anomaly"].mean()*100
               if len(df_e[df_e["season"]==s])>0 else 0 for s in seasons]
    counts = [int(df_e[df_e["season"]==s]["anomaly"].sum()) for s in seasons]
    totals = [len(df_e[df_e["season"]==s]) for s in seasons]
    bars   = axes[0].bar(seasons, rates, color=colors, alpha=0.8, edgecolor="white")
    for bar, r, c, t in zip(bars, rates, counts, totals):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                     f"{c}/{t}\n({r:.1f}%)", ha="center", va="bottom", fontsize=9)
    axes[0].set_title("Anomaly Rate by Season", fontweight="bold")
    axes[0].set_ylabel("Anomaly rate (%)"); axes[0].grid(alpha=0.3, axis="y")
    mean_errs = [df_e[df_e["season"]==s]["error"].mean() for s in seasons]
    bars2 = axes[1].bar(seasons, mean_errs, color=colors, alpha=0.8, edgecolor="white")
    for bar, v in zip(bars2, mean_errs):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.00001,
                     f"{v:.5f}", ha="center", va="bottom", fontsize=9)
    axes[1].set_title("Mean Reconstruction Error by Season", fontweight="bold")
    axes[1].set_ylabel("Mean MSE"); axes[1].grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/04_season_breakdown.png", dpi=150)
    plt.close(); print("  ✓ 04_season_breakdown.png")

def plot_threshold_sensitivity(test_errs, threshold):
    pcts       = list(range(80, 100))
    thresholds = [np.percentile(test_errs, p) for p in pcts]
    rates      = [(test_errs > t).mean()*100 for t in thresholds]
    fig, ax1   = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()
    ax1.plot(pcts, thresholds, color=BLUE, lw=2, marker="o", ms=4,
             label="Threshold value")
    ax2.plot(pcts, rates, color=RED, lw=2, marker="s", ms=4,
             label="Anomaly rate (%)")
    ax1.axvline(95, color="gray", ls=":", lw=1.5,
                label=f"95th pct  (floor = {threshold:.5f})")
    ax1.set_xlabel("Threshold percentile (of test errors)")
    ax1.set_ylabel("Threshold MSE", color=BLUE)
    ax2.set_ylabel("Anomaly rate (%)", color=RED)
    ax1.set_title("Threshold Sensitivity Analysis", fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, fontsize=9)
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/05_threshold_sensitivity.png", dpi=150)
    plt.close(); print("  ✓ 05_threshold_sensitivity.png")

# ── MAIN ──────────────────────────────────────────────────────
def run():
    print("=" * 60)
    print(f"  STEP 3 — Anomaly Detection  [{DEVICE}]")
    print("=" * 60)

    # Deterministic mode for reproducible inference on GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print(f"  (deterministic-algorithms not fully available: {e})")

    ckpt = torch.load(f"{MODEL_DIR}/best_model.pt",
                      map_location=DEVICE, weights_only=False)
    model = LSTMAutoencoder().to(DEVICE)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    model_threshold = float(np.load(f"{MODEL_DIR}/threshold.npy"))
    print(f"  Model threshold (95th pct CLEAN val errors, step2, reference "
          f"only): {model_threshold:.6f}")

    df = pd.read_csv(RAW_CSV, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    energy  = df["energy_kwh"].values.astype(np.float32)
    n_total = len(df)

    # Test = trailing pool_end..end (same formula as step2 — zero overlap guaranteed).
    pool_end  = int(n_total * TRAIN_RATIO) + int(n_total * VAL_RATIO)
    test_mask = np.zeros(n_total, dtype=bool)
    test_mask[pool_end:] = True
    lo = max(0, pool_end - (WINDOW - 1))
    hi = min(n_total, pool_end + (WINDOW - 1))
    test_mask[lo:hi] = False   # gap around the pool/test boundary

    # Val = exact blocked/stratified set step2 selected (loaded, not re-derived).
    val_mask = np.load(f"{MODEL_DIR}/val_mask.npy")
    if len(val_mask) != n_total:
        raise ValueError(
            f"val_mask.npy length ({len(val_mask)}) != etl_raw.csv rows "
            f"({n_total}) - re-run step2_train.py to regenerate it.")

    scaled = scaler.transform(energy.reshape(-1,1)).flatten()
    X_test, idx_test = make_sequences_from_mask(scaled, test_mask, WINDOW)
    ts_test = df["datetime"].values[idx_test]

    print(f"  Test sequences   : {len(X_test):,}  (raw — anomalies preserved)")
    print(f"  Test date range  : {pd.to_datetime(ts_test[0]).date()} → "
          f"{pd.to_datetime(ts_test[-1]).date()}")

    test_errs, test_recon = get_errors_recon(model, X_test)
    act_te = inv(X_test[:,-1,0], scaler)
    rec_te = inv(test_recon[:,-1], scaler)
    mv_te  = compute_metrics(act_te, rec_te)

    print(f"\n  Test reconstruction (raw data):")
    print(f"  MAE={mv_te['mae']:.4f} kWh  RMSE={mv_te['rmse']:.4f}  "
          f"MAPE={mv_te['mape']:.2f}%  R²={mv_te['r2']:.4f}")

    # Val errors → point + smooth signal floors
    X_val_seq, _ = make_sequences_from_mask(scaled, val_mask, WINDOW)
    val_errs, _  = get_errors_recon(model, X_val_seq)
    val_smooth   = pd.Series(val_errs).rolling(SMOOTH_WIN, min_periods=1).mean().values
    # Both floors on RAW val errors — same domain as test_errs.
    threshold    = float(np.percentile(val_errs, FLOOR_PCT))
    smooth_floor = float(np.percentile(val_smooth, FLOOR_PCT))
    print(f"  Point floor  ({FLOOR_PCT}th pct RAW val errors): {threshold:.6f}")
    print(f"  Smooth floor ({FLOOR_PCT}th pct smoothed RAW val): {smooth_floor:.6f}")

    # ── Dual-signal detection ─────────────────────────────────
    roll_thresh = rolling_threshold(test_errs, k=POINT_K,
                                    window=THRESH_WIN, lag=THRESH_LAG,
                                    floor=threshold)
    mask_point  = test_errs > roll_thresh

    smooth_errs   = pd.Series(test_errs).rolling(SMOOTH_WIN, min_periods=1).mean().values
    smooth_thresh = rolling_threshold(smooth_errs, k=SMOOTH_K,
                                      window=THRESH_WIN, lag=THRESH_LAG,
                                      floor=smooth_floor)
    mask_smooth   = smooth_errs > smooth_thresh

    mask   = mask_point | mask_smooth
    events = group_events(ts_test, mask)
    events = [e for e in events if e[2] >= MIN_EVENT_H]  # drop noise

    # Rebuild masks from filtered events — counts, dots and CSV all consistent
    ts_dt      = pd.to_datetime(ts_test)
    event_mask = np.zeros(len(ts_test), dtype=bool)
    for start, end, dur in events:
        event_mask |= (ts_dt >= start) & (ts_dt <= end)
    mask_point  = mask_point  & event_mask
    mask_smooth = mask_smooth & event_mask
    mask        = event_mask

    print(f"\n  Flagged (point)  : {mask_point.sum():,}  "
          f"({mask_point.mean()*100:.2f}%)  [post MIN_EVENT_H filter]")
    print(f"  Flagged (smooth) : {mask_smooth.sum():,}  "
          f"({mask_smooth.mean()*100:.2f}%)  [post MIN_EVENT_H filter]")
    print(f"  Flagged (OR)     : {mask.sum():,}  "
          f"({mask.mean()*100:.2f}%)  [post MIN_EVENT_H filter]")
    print(f"  Distinct events  : {len(events)}")

    if events:
        ts_dt = pd.to_datetime(ts_test)
        print(f"\n  {'Start':<22} {'End':<22} {'Dur':>5}  {'MaxMSE':>10}  "
              f"{'Margin':>7}  {'Signal':>6}  {'Sustained':>9}")
        print(f"  {'-'*22} {'-'*22} {'-'*5}  {'-'*10}  "
              f"{'-'*7}  {'-'*6}  {'-'*9}")
        event_rows = []
        for start, end, dur in sorted(events, key=lambda x: -x[2]):
            in_ev    = (ts_dt >= start) & (ts_dt <= end)
            max_e    = test_errs[in_ev].max() if in_ev.any() else float("nan")
            max_e_sm = smooth_errs[in_ev].max() if in_ev.any() else float("nan")
            fires_pt = bool(mask_point[in_ev].any())
            fires_sm = bool(mask_smooth[in_ev].any())
            signal   = "BOTH" if (fires_pt and fires_sm) else ("PT" if fires_pt else "SM")
            # Margin: peak error above the floor of the signal(s) that fired.
            # BOTH events take the more conservative (lower) of the two margins.
            margin_pt = (max_e - threshold) / threshold
            margin_sm = (max_e_sm - smooth_floor) / smooth_floor
            if signal == "PT":
                margin = margin_pt
            elif signal == "SM":
                margin = margin_sm
            else:
                margin = min(margin_pt, margin_sm)
            # Sustained: fraction of event window where a raw per-hour flag fired.
            raw_hits  = (mask_point | mask_smooth)[in_ev]
            sustained = raw_hits.mean() if in_ev.any() else float("nan")
            print(f"  {str(start):<22} {str(end):<22} "
                  f"{dur:>4}h  {max_e:>10.6f}  "
                  f"{margin:>7.2f}x  {signal:>6}  {sustained:>8.1%}")
            event_rows.append(dict(start=start, end=end, duration_hours=dur,
                                   max_mse=max_e, margin=round(margin, 4),
                                   signal=signal, sustained=round(sustained, 4)))
        print()
        print("  Margin  = (MaxMSE − floor) / floor  — lower → less confident")
        print("  Signal  = PT (point only) | SM (smooth only) | BOTH")
        print("  Sustained = fraction of event window where combined mask is raised")

        low_conf = [r for r in event_rows if r["margin"] < LOW_CONF_MARGIN]
        if low_conf:
            print(f"\n  ⚠  {len(low_conf)} low-confidence event(s) "
                  f"(margin < {LOW_CONF_MARGIN}x) — verify against operational records:")
            for r in low_conf:
                print(f"     {r['start']}  margin={r['margin']:.2f}x  signal={r['signal']}")

    pd.DataFrame({
        "timestamp":         pd.to_datetime(ts_test),
        "actual_kwh":        act_te,
        "reconstructed_kwh": rec_te,
        "residual_kwh":      act_te - rec_te,
        "rec_error_mse":     test_errs,
        "smoothed_error":    smooth_errs,
        "point_threshold":   roll_thresh,
        "smooth_threshold":  smooth_thresh,
        "anomaly_point":     mask_point,
        "anomaly_smooth":    mask_smooth,
        "anomaly":           mask,
    }).to_csv(f"{RESULT_DIR}/anomaly_results.csv", index=False)

    pd.DataFrame(events, columns=["start","end","duration_hours"]
                 ).to_csv(f"{RESULT_DIR}/anomaly_events.csv", index=False) \
        if not events else \
        pd.DataFrame(event_rows).to_csv(f"{RESULT_DIR}/anomaly_events.csv", index=False)

    print("\n  Generating detection plots...")
    plot_error_timeline(ts_test, test_errs, smooth_errs,
                        roll_thresh, smooth_thresh,
                        mask_point, mask_smooth, mask, threshold, smooth_floor)
    plot_event_calendar(events, ts_test)
    plot_test_reconstruction(ts_test, act_te, rec_te, mask)
    plot_season_breakdown(df, ts_test, test_errs, mask)
    plot_threshold_sensitivity(test_errs, threshold)

    print(f"\n  ✅ Results → {RESULT_DIR}/anomaly_results.csv")
    print(f"  ✅ Events  → {RESULT_DIR}/anomaly_events.csv")
    print(f"  ✅ Plots   → {PLOT_DIR}/")

if __name__ == "__main__":
    run()