"""
step3_detect_water.py — Anomaly Detection (WATER, FIXED threshold)
Input : etl_output/etl_raw.csv, model/
Output: results/anomaly_results.csv, results/anomaly_events.csv, plots/detect/
Dual-signal fixed threshold (no rolling/adaptive component — water assumed stable):
  point  : MSE > mean(val_errs) + POINT_K·std(val_errs)
  smooth : 4h rolling mean of MSE > mean(smooth(val_errs)) + SMOOTH_K·std
  combined: OR of both; consecutive flags within GAP_H merged into events.
Threshold calibrated on VAL errors (out-of-sample estimate), not TRAIN.
Run: python step3_detect_water.py
"""

import sys, os, warnings, pickle
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

# Must be set before torch initialises to make cuBLAS GEMMs deterministic.
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

from config import (RAW_CSV, MODEL_DIR, DETECT_PLOT_DIR as PLOT_DIR,
                     RESULT_DIR)

# ── CONFIG ────────────────────────────────────────────────────
WINDOW      = 24
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# Must match step2_train_water.py exactly.
HOLD_OUT_EVERY_N_HOURS = 168 * 6
HOLD_OUT_LEN           = 168
HIDDEN_DIM  = 32
LATENT_DIM  = 12
GAP_H       = 6
MIN_EVENT_H = 2     # drop single-hour noise flags
SMOOTH_WIN  = 4

POINT_K     =   # tuned via tune_thresholds_water.py
SMOOTH_K    =   # tuned via tune_thresholds_water.py
LOW_CONF_MARGIN = 2.0  # events with margin < this are flagged for manual review

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

def build_pool_masks(n_total, window, train_ratio, val_ratio,
                      hold_out_every, hold_out_len):
    """Same scheme as step2_train_water.py: reserve trailing pool_end..end as TEST;
    hold out 1 week per hold_out_every hours as val, with a window-1 gap at boundaries."""
    pool_end = int(n_total * train_ratio) + int(n_total * val_ratio)

    is_val = np.zeros(n_total, dtype=bool)
    pos = 0
    while pos < pool_end:
        val_start = pos + hold_out_every - hold_out_len
        val_end   = pos + hold_out_every
        if val_start < pool_end:
            is_val[val_start:min(val_end, pool_end)] = True
        pos += hold_out_every

    pool_mask = np.zeros(n_total, dtype=bool)
    pool_mask[:pool_end] = True

    gap_mask = np.zeros(n_total, dtype=bool)
    edges = np.where(np.diff(is_val.astype(int)) != 0)[0]
    for e in edges:
        lo = max(0, e - (window - 1) + 1)
        hi = min(n_total, e + (window - 1) + 1)
        gap_mask[lo:hi] = True
    lo = max(0, pool_end - (window - 1))
    hi = min(n_total, pool_end + (window - 1))
    gap_mask[lo:hi] = True

    train_mask = pool_mask & (~is_val) & (~gap_mask)
    val_mask   = is_val & pool_mask & (~gap_mask)

    return train_mask, val_mask, gap_mask

def make_sequences_from_mask(values, valid_mask, window):
    """Only sequences fully contained within a contiguous valid_mask region."""
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

def fixed_threshold(baseline_errors, k):
    """Fixed threshold: mean + k·std of baseline errors. Applied flat — no rolling window."""
    baseline_errors = np.asarray(baseline_errors, dtype=np.float64)
    return float(baseline_errors.mean() + k * baseline_errors.std())

def group_events(ts, mask, gap_h=GAP_H):
    """Merge consecutive flagged timestamps into events. Gap ≤ gap_h hours between flags → same event."""
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
    mask = actual > 10  # 10 L/h == the old 0.01 m3/h floor
    return dict(
        mae  = mean_absolute_error(actual, predicted),
        rmse = np.sqrt(mean_squared_error(actual, predicted)),
        r2   = r2_score(actual, predicted),
        mape = (np.mean(np.abs((actual[mask]-predicted[mask])/actual[mask]))*100
                if mask.any() else float("nan")),
    )

# ── PLOTS ─────────────────────────────────────────────────────
def plot_error_timeline(ts, errors, smooth_err,
                        point_thresh, smooth_thresh,
                        mask_pt, mask_sm, mask):
    ts_ = pd.to_datetime(ts)
    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
    fig.suptitle(
        f"Dual-Signal Anomaly Detection — Test Set (raw data, anomalies preserved)\n"
        f"Combined (OR): {mask.sum()} flagged  |  "
        f"Point: {mask_pt.sum()}  |  Smooth: {mask_sm.sum()}",
        fontweight="bold", y=1.01, fontsize=13)

    axes[0].plot(ts_, errors, color=BLUE, lw=0.7, alpha=0.85,
                 label="Reconstruction error (MSE)")
    axes[0].axhline(point_thresh, color=RED, lw=1.5, ls="--",
                 label=f"Fixed threshold")
    axes[0].scatter(ts_[mask_pt], errors[mask_pt], color="darkred",
                    s=20, zorder=5, label=f"Flagged ({mask_pt.sum()})")
    axes[0].set_title(
        f"Reconstruction error vs FIXED threshold (k={POINT_K})",
        fontweight="bold")
    axes[0].set_ylabel("MSE"); axes[0].legend(fontsize=9); axes[0].grid(alpha=0.25)

    axes[1].plot(ts_, smooth_err, color=GRN, lw=0.9,
                 label=f"Smoothed error")
    axes[1].axhline(smooth_thresh, color=ORG, lw=1.5, ls="--",
                 label=f"Fixed threshold")
    axes[1].scatter(ts_[mask_sm], smooth_err[mask_sm], color="saddlebrown",
                    s=20, zorder=5, label=f"Flagged ({mask_sm.sum()})")
    axes[1].set_title(
        f"Smoothed error vs FIXED threshold (k={SMOOTH_K})",
        fontweight="bold")
    axes[1].set_ylabel("Smoothed MSE"); axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.25)
    axes[1].set_xlim(ts_[0], ts_[-1])   # clip to the test window (no padding past dataset end)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))  # ticks on Mondays
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha="right")
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
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))  # ticks on Mondays
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax.set_xlim(mdates.date2num(pd.to_datetime(ts_range[0])),
                    mdates.date2num(pd.to_datetime(ts_range[-1])))
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
    fig, ax = plt.subplots(figsize=(18, 5))
    fig.suptitle(
        f"Test Set — Actual vs Reconstructed (first {n_days} days)\n"
        f"[Raw data — real anomalies preserved for detection]",
        fontweight="bold", fontsize=13)
    ORANGE = "#FF6D00"
    ax.plot(ts_, act, color=BLUE,   lw=1.3, label="Actual (raw)")
    ax.plot(ts_, rec, color=ORANGE, lw=1.3, ls="--",
            label="Reconstructed (expected normal)")
    ax.scatter(ts_[m], act[m], color="darkred", s=25, zorder=5,
               label=f"Anomaly ({m.sum()} pts)")
    ax.set_ylabel("Consumption (L/h)"); ax.set_xlabel("Date")
    ax.legend(); ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/03_test_reconstruction.png", dpi=150)
    plt.close(); print("  ✓ 03_test_reconstruction.png")

def plot_hourly_breakdown(ts, errors, mask):
    """Anomaly rate and mean error by hour_of_day (replaces per-season breakdown — no season context here)."""
    ts_ = pd.to_datetime(ts)
    df_e = pd.DataFrame({"hour": ts_.hour, "error": errors, "anomaly": mask})
    hours = list(range(24))
    rates  = [df_e[df_e["hour"]==h]["anomaly"].mean()*100
              if len(df_e[df_e["hour"]==h])>0 else 0 for h in hours]
    counts = [int(df_e[df_e["hour"]==h]["anomaly"].sum()) for h in hours]
    mean_errs = [df_e[df_e["hour"]==h]["error"].mean() for h in hours]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("Per-Hour-of-Day Analysis (test set — raw data)",
                 fontweight="bold", fontsize=13)
    bars = axes[0].bar(hours, rates, color=BLUE, alpha=0.8, edgecolor="white")
    for bar, r, c in zip(bars, rates, counts):
        if c > 0:
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                         f"{c}", ha="center", va="bottom", fontsize=7)
    axes[0].set_title("Anomaly Rate by Hour of Day", fontweight="bold")
    axes[0].set_xlabel("Hour of day"); axes[0].set_ylabel("Anomaly rate (%)")
    axes[0].set_xticks(hours); axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(hours, mean_errs, color=GRN, alpha=0.8, edgecolor="white")
    axes[1].set_title("Mean Reconstruction Error by Hour of Day", fontweight="bold")
    axes[1].set_xlabel("Hour of day"); axes[1].set_ylabel("Mean MSE")
    axes[1].set_xticks(hours); axes[1].grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/04_hourly_breakdown.png", dpi=150)
    plt.close(); print("  ✓ 04_hourly_breakdown.png")

def plot_threshold_sensitivity(test_errs, point_thresh):
    pcts       = list(range(80, 100))
    thresholds = [np.percentile(test_errs, p) for p in pcts]
    rates      = [(test_errs > t).mean()*100 for t in thresholds]
    fig, ax1   = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()
    ax1.plot(pcts, thresholds, color=BLUE, lw=2, marker="o", ms=4,
             label="Threshold value (if set by test-error percentile)")
    ax2.plot(pcts, rates, color=RED, lw=2, marker="s", ms=4,
             label="Anomaly rate (%)")
    ax1.axhline(point_thresh, color="gray", ls=":", lw=1.5,
                label=f"Actual FIXED threshold used = {point_thresh:.5f}")
    ax1.set_xlabel("Threshold percentile (of test errors, for reference only)")
    ax1.set_ylabel("Threshold MSE", color=BLUE)
    ax2.set_ylabel("Anomaly rate (%)", color=RED)
    ax1.set_title("Threshold Sensitivity Analysis", fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, fontsize=8)
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/05_threshold_sensitivity.png", dpi=150)
    plt.close(); print("  ✓ 05_threshold_sensitivity.png")

# ── MAIN ──────────────────────────────────────────────────────
def run():
    print("=" * 60)
    print(f"  STEP 3 — Anomaly Detection (FIXED threshold)  [{DEVICE}]")
    print("=" * 60)

    # Deterministic mode for reproducible detection output.
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

    df = pd.read_csv(RAW_CSV, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    water   = df["consumption_l"].values.astype(np.float32)
    n_total = len(df)
    pool_end = int(n_total * TRAIN_RATIO) + int(n_total * VAL_RATIO)

    scaled         = scaler.transform(water.reshape(-1,1)).flatten()
    X_all, idx_all = make_sequences(scaled, WINDOW)
    ts_all         = df["datetime"].values[idx_all]

    # Interleaved val mask — same as step2_train_water.py. Calibrate on VAL, not TRAIN.
    train_mask, val_mask, _ = build_pool_masks(
        n_total, WINDOW, TRAIN_RATIO, VAL_RATIO,
        HOLD_OUT_EVERY_N_HOURS, HOLD_OUT_LEN)
    X_val, _ = make_sequences_from_mask(scaled, val_mask, WINDOW)

    # WINDOW-1 gap at pool/test boundary: every test window lies entirely in test.
    test_start_k = pool_end
    X_test  = X_all[test_start_k:]
    ts_test = ts_all[test_start_k:]

    print(f"  Test sequences   : {len(X_test):,}  (raw — anomalies preserved)")
    print(f"  Test date range  : {pd.to_datetime(ts_test[0]).date()} → "
          f"{pd.to_datetime(ts_test[-1]).date()}")

    test_errs, test_recon = get_errors_recon(model, X_test)
    act_te = inv(X_test[:,-1,0], scaler)
    rec_te = inv(test_recon[:,-1], scaler)
    mv_te  = compute_metrics(act_te, rec_te)

    print(f"\n  Test reconstruction (raw data):")
    print(f"  MAE={mv_te['mae']:.2f} L/h  RMSE={mv_te['rmse']:.4f}  "
          f"MAPE={mv_te['mape']:.2f}%  R²={mv_te['r2']:.4f}")

    # Fixed thresholds from VAL errors (never used for gradient updates → fair normal-error estimate).
    val_errs        = get_errors_recon(model, X_val)[0]
    point_thresh    = fixed_threshold(val_errs, k=POINT_K)
    val_smooth      = pd.Series(val_errs).rolling(SMOOTH_WIN, min_periods=1).mean().values
    smooth_thresh   = fixed_threshold(val_smooth, k=SMOOTH_K)
    print(f"\n  Fixed point threshold  (val mean+{POINT_K}σ)  : {point_thresh:.6f}")
    print(f"  Fixed smooth threshold (val mean+{SMOOTH_K}σ) : {smooth_thresh:.6f}")

    # Dual-signal detection
    mask_point  = test_errs > point_thresh

    smooth_errs = pd.Series(test_errs).rolling(SMOOTH_WIN, min_periods=1).mean().values
    mask_smooth = smooth_errs > smooth_thresh

    mask   = mask_point | mask_smooth
    events = group_events(ts_test, mask)
    events = [e for e in events if e[2] >= MIN_EVENT_H]  # drop noise

    # Keep raw per-hour flags before event-fill to compute genuine sustained fraction below.
    mask_point_raw  = mask_point.copy()
    mask_smooth_raw = mask_smooth.copy()
    mask_raw        = mask.copy()

    # Rebuild masks from filtered events for consistent counts/plots/CSV
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

    event_rows = []
    if events:
        ts_dt = pd.to_datetime(ts_test)
        print(f"\n  {'Start':<22} {'End':<22} {'Dur':>5}  {'MaxMSE':>10}  "
              f"{'Margin':>7}  {'Signal':>6}  {'Sustained':>9}")
        print(f"  {'-'*22} {'-'*22} {'-'*5}  {'-'*10}  "
              f"{'-'*7}  {'-'*6}  {'-'*9}")
        for start, end, dur in sorted(events, key=lambda x: -x[2]):
            in_ev    = (ts_dt >= start) & (ts_dt <= end)
            max_e    = test_errs[in_ev].max() if in_ev.any() else float("nan")
            max_sm   = smooth_errs[in_ev].max() if in_ev.any() else float("nan")
            fires_pt = bool(mask_point_raw[in_ev].any())
            fires_sm = bool(mask_smooth_raw[in_ev].any())
            signal   = "BOTH" if (fires_pt and fires_sm) else ("PT" if fires_pt else "SM")

            # Margin against whichever threshold(s) fired; stronger of the two when BOTH.
            margin_pt = (max_e  - point_thresh)  / point_thresh
            margin_sm = (max_sm - smooth_thresh) / smooth_thresh
            if fires_pt and fires_sm:
                margin = max(margin_pt, margin_sm)
            elif fires_pt:
                margin = margin_pt
            else:
                margin = margin_sm

            # Fraction of event window individually flagged by raw signal (not just bridged by gap rule).
            sustained = mask_raw[in_ev].mean() if in_ev.any() else float("nan")
            print(f"  {str(start):<22} {str(end):<22} "
                  f"{dur:>4}h  {max_e:>10.6f}  "
                  f"{margin:>7.2f}x  {signal:>6}  {sustained:>8.1%}")
            event_rows.append(dict(start=start, end=end, duration_hours=dur,
                                   max_mse=max_e, margin=round(margin, 4),
                                   signal=signal, sustained=round(sustained, 4)))
        print()
        print("  Margin  = confidence margin above whichever threshold(s) fired")
        print("            (point and/or smooth; stronger of the two when BOTH) — lower → less confident")
        print("  Signal  = PT (point only) | SM (smooth only) | BOTH")
        print("  Sustained = fraction of event window individually flagged by the raw")
        print("              signal (not hours only bridged by the merge-gap rule)")

        low_conf = [r for r in event_rows if r["margin"] < LOW_CONF_MARGIN]
        if low_conf:
            print(f"\n  ⚠  {len(low_conf)} low-confidence event(s) "
                  f"(margin < {LOW_CONF_MARGIN}x) — verify against operational records:")
            for r in low_conf:
                print(f"     {r['start']}  margin={r['margin']:.2f}x  signal={r['signal']}")

    pd.DataFrame({
        "timestamp":         pd.to_datetime(ts_test),
        "actual_l":          act_te,
        "reconstructed_l":   rec_te,
        "residual_l":        act_te - rec_te,
        "rec_error_mse":     test_errs,
        "smoothed_error":    smooth_errs,
        "point_threshold":   point_thresh,
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
                        point_thresh, smooth_thresh,
                        mask_point, mask_smooth, mask)
    plot_event_calendar(events, ts_test)
    plot_test_reconstruction(ts_test, act_te, rec_te, mask)
    plot_hourly_breakdown(ts_test, test_errs, mask)
    plot_threshold_sensitivity(test_errs, point_thresh)

    print(f"\n  ✅ Results → {RESULT_DIR}/anomaly_results.csv")
    print(f"  ✅ Events  → {RESULT_DIR}/anomaly_events.csv")
    print(f"  ✅ Plots   → {PLOT_DIR}/")

if __name__ == "__main__":
    run()