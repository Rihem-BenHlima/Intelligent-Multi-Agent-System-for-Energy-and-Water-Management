"""
tune_thresholds.py — Threshold Parameter Selection via Elbow Method
====================================================================
Selects optimal POINT_K and SMOOTH_K without ground truth labels.

Methodology (label-free):
  For each k value in a range, compute:
    1. Flagged rate  : fraction of test sequences flagged as anomalous
    2. Event count   : number of distinct anomaly events detected
  The optimal k sits at the ELBOW of both curves — where increasing k
  further produces diminishing reduction in false alarms.

Split/floor methodology matches step2_train.py / step3_detect.py exactly:
  - test = trailing pool_end..end of RAW data (same pool_end formula,
    gap-buffered), never touched during training
  - val  = model/val_mask.npy (the blocked/stratified set step2 selected),
    evaluated on RAW data — same domain as test_errs, not CLEAN data
  - both floors = FLOOR_PCT-th pct of val errors, same percentile,
    each on its own signal's distribution (raw vs smoothed)

Output:
  plots/tune/elbow_combined.png
  Prints recommended POINT_K and SMOOTH_K to copy into step3_detect.py.

Run:  python tune_thresholds.py
"""

import sys, os, warnings, pickle
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import RAW_CSV, MODEL_DIR, PLOTS_TUNE_DIR as PLOT_DIR

# ── CONFIG (must match step2_train.py and step3_detect.py) ────
WINDOW      = 24
TRAIN_RATIO = 0.70   # only used to locate pool_end - identical formula to
VAL_RATIO   = 0.15   # step2/step3, so test boundary agrees exactly
HIDDEN_DIM  = 32
LATENT_DIM  = 12
THRESH_WIN  = 168
THRESH_LAG  = 4
SMOOTH_WIN  = 4
GAP_H       = 6       # hours gap to merge consecutive flags into one event
MIN_EVENT_H = 2       # minimum event duration to count (drops noise)
FLOOR_PCT   = 80      # matches step3_detect.py; lets rolling term drive threshold in calm periods
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

K_RANGE     = np.arange(2.0, 8.5, 0.5)   # k values to evaluate

os.makedirs(PLOT_DIR, exist_ok=True)

# ── MODEL ─────────────────────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1   = nn.LSTM(1,          HIDDEN_DIM, batch_first=True)
        self.enc2   = nn.LSTM(HIDDEN_DIM, LATENT_DIM, batch_first=True)
        self.dec1   = nn.LSTM(LATENT_DIM, HIDDEN_DIM, batch_first=True)
        self.output = nn.Linear(HIDDEN_DIM, 1)
    def forward(self, x):
        o,_=self.enc1(x); o,_=self.enc2(o)
        l=o[:,-1:,:].repeat(1,x.size(1),1); o,_=self.dec1(l)
        return self.output(o)

# ── HELPERS ───────────────────────────────────────────────────
def make_sequences_from_mask(values, valid_mask, window):
    """Only sequences fully contained within a contiguous valid_mask
    region — mirrors step2_train.py/step3_detect.py's helper of the
    same purpose."""
    X, idx = [], []
    for i in range(len(values) - window + 1):
        if valid_mask[i:i + window].all():
            X.append(values[i:i + window]); idx.append(i + window - 1)
    return np.array(X, dtype=np.float32)[:, :, np.newaxis], np.array(idx)

def get_errors(model, X):
    loader = DataLoader(TensorDataset(torch.tensor(X)), batch_size=512, shuffle=False)
    errs = []
    with torch.no_grad():
        for (b,) in loader:
            b = b.to(DEVICE); r = model(b)
            errs.append(((r - b)**2).mean(dim=[1,2]).cpu().numpy())
    return np.concatenate(errs)

def rolling_threshold(errors, k, window=THRESH_WIN, lag=THRESH_LAG, floor=None, seed=None):
    """Identical to step3_detect.py."""
    errors = np.asarray(errors, dtype=np.float64)
    if seed is not None:
        seed = np.asarray(seed, dtype=np.float64)
        n_seed = len(seed)
        errors = np.concatenate([seed, errors])
    else:
        n_seed = 0
    clip_val = np.percentile(errors[n_seed:], 95)
    eb = np.clip(errors, 0, clip_val)
    s  = pd.Series(eb)
    rm = s.rolling(window, min_periods=1).mean()
    rs = s.rolling(window, min_periods=1).std().fillna(0)
    thresh = (rm.shift(lag).fillna(rm.iloc[0]) +
              k * rs.shift(lag).fillna(rs.iloc[0])).values[n_seed:]
    if floor is not None:
        thresh = np.maximum(thresh, floor)
    return thresh

def count_events(ts, mask, gap_h=GAP_H, min_h=MIN_EVENT_H):
    """Count distinct events from a boolean mask (same logic as step3_detect.py)."""
    times = pd.to_datetime(ts[mask])
    if len(times) == 0:
        return 0
    events, s, e = [], times[0], times[0]
    for t in times[1:]:
        if (t - e).total_seconds() / 3600 <= gap_h:
            e = t
        else:
            dur = int((e - s).total_seconds() / 3600) + 1
            if dur >= min_h:
                events.append(dur)
            s = e = t
    dur = int((e - s).total_seconds() / 3600) + 1
    if dur >= min_h:
        events.append(dur)
    return len(events)

def find_elbow(k_vals, curve):
    """Elbow via maximum perpendicular distance from the line connecting
    the first and last points of the curve."""
    p1 = np.array([k_vals[0],  curve[0]])
    p2 = np.array([k_vals[-1], curve[-1]])
    line = p2 - p1
    line_norm = line / np.linalg.norm(line)
    dists = []
    for i, k in enumerate(k_vals):
        p = np.array([k, curve[i]])
        proj = p1 + np.dot(p - p1, line_norm) * line_norm
        dists.append(np.linalg.norm(p - proj))
    return int(np.argmax(dists))

def plot_elbow(k_vals, flag_rates, event_counts, elbow_idx, signal_name, fname):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        f"Elbow Method — {signal_name}\n"
        f"Optimal k = {k_vals[elbow_idx]:.1f}  "
        f"(flagged rate={flag_rates[elbow_idx]*100:.2f}%,  "
        f"events={event_counts[elbow_idx]})",
        fontweight="bold", fontsize=12)

    color_rate   = "#1565C0"
    color_events = "#E53935"

    ax1.plot(k_vals, flag_rates * 100, color=color_rate, lw=2,
             marker="o", ms=5, label="Flagged rate (%)")
    ax1.axvline(k_vals[elbow_idx], color="gray", ls="--", lw=1.5,
                label=f"Elbow  k={k_vals[elbow_idx]:.1f}")
    ax1.scatter([k_vals[elbow_idx]], [flag_rates[elbow_idx]*100],
                color="gold", s=120, zorder=6, edgecolors="black", lw=1.2)
    ax1.set_xlabel("k  (threshold = mean + k·σ)")
    ax1.set_ylabel("Flagged rate (%)", color=color_rate)
    ax1.tick_params(axis="y", labelcolor=color_rate)
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(k_vals, event_counts, color=color_events, lw=2,
             marker="s", ms=5, ls="--", label="Event count")
    ax2.scatter([k_vals[elbow_idx]], [event_counts[elbow_idx]],
                color="gold", s=120, zorder=6, edgecolors="black", lw=1.2)
    ax2.set_ylabel("Distinct event count", color=color_events)
    ax2.tick_params(axis="y", labelcolor=color_events)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {fname}")

# ── MAIN ──────────────────────────────────────────────────────
def run():
    print("=" * 60)
    print("  THRESHOLD TUNING — Elbow Method (no ground truth)")
    print("=" * 60)

    ckpt  = torch.load(f"{MODEL_DIR}/best_model.pt", map_location=DEVICE, weights_only=False)
    model = LSTMAutoencoder().to(DEVICE)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    df = pd.read_csv(RAW_CSV, parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
    energy  = df["energy_kwh"].values.astype(np.float32)
    n_total = len(df)
    scaled  = scaler.transform(energy.reshape(-1, 1)).flatten()

    # Test = trailing pool_end..end, gap-buffered. Val = model/val_mask.npy.
    pool_end  = int(n_total * TRAIN_RATIO) + int(n_total * VAL_RATIO)
    test_mask = np.zeros(n_total, dtype=bool)
    test_mask[pool_end:] = True
    lo = max(0, pool_end - (WINDOW - 1))
    hi = min(n_total, pool_end + (WINDOW - 1))
    test_mask[lo:hi] = False

    val_mask = np.load(f"{MODEL_DIR}/val_mask.npy")
    if len(val_mask) != n_total:
        raise ValueError(
            f"val_mask.npy length ({len(val_mask)}) != {RAW_CSV} rows "
            f"({n_total}) - re-run step2_train.py to regenerate it.")

    X_test, idx_test = make_sequences_from_mask(scaled, test_mask, WINDOW)
    ts_test = pd.to_datetime(df["datetime"].values[idx_test])
    test_errs   = get_errors(model, X_test)
    smooth_errs = pd.Series(test_errs).rolling(SMOOTH_WIN, min_periods=1).mean().values
    print(f"  Test sequences : {len(X_test):,}")
    print(f"  Test range     : {ts_test[0].date()} → {ts_test[-1].date()}")

    X_val, _  = make_sequences_from_mask(scaled, val_mask, WINDOW)
    val_errs  = get_errors(model, X_val)
    val_smooth = pd.Series(val_errs).rolling(SMOOTH_WIN, min_periods=1).mean().values

    # Floors: FLOOR_PCT-th pct of RAW val errors, per signal's own distribution.
    threshold    = float(np.percentile(val_errs, FLOOR_PCT))
    smooth_floor = float(np.percentile(val_smooth, FLOOR_PCT))
    print(f"  Floor (point)  : {threshold:.6f}")
    print(f"  Floor (smooth) : {smooth_floor:.6f}")
    print("-" * 60)

    # ── Joint grid over (k_pt, k_sm) — combined OR mask ─────
    print(f"\n  {'k_pt':>5} {'k_sm':>5} {'OR_rate':>9} {'OR_events':>10}")
    print(f"  {'-'*5} {'-'*5} {'-'*9} {'-'*10}")

    combined_flag_rates  = []
    combined_event_counts= []
    k_pairs              = []

    for k_pt in K_RANGE:
        for k_sm in K_RANGE:
            t_pt = rolling_threshold(test_errs,   k_pt, floor=threshold)
            t_sm = rolling_threshold(smooth_errs, k_sm, floor=smooth_floor)
            m_or = (test_errs > t_pt) | (smooth_errs > t_sm)
            rate = m_or.mean()
            evts = count_events(ts_test.values, m_or)
            combined_flag_rates.append(rate)
            combined_event_counts.append(evts)
            k_pairs.append((k_pt, k_sm))

    combined_flag_rates   = np.array(combined_flag_rates)
    combined_event_counts = np.array(combined_event_counts, dtype=float)

    diag_idx   = [i for i,(kp,ks) in enumerate(k_pairs) if kp == ks]
    diag_k     = np.array([k_pairs[i][0] for i in diag_idx])
    diag_rates = combined_flag_rates[diag_idx]
    diag_evts  = combined_event_counts[diag_idx]

    for i in diag_idx:
        kp, ks = k_pairs[i]
        print(f"  {kp:>5.1f} {ks:>5.1f} {combined_flag_rates[i]*100:>8.2f}% {int(combined_event_counts[i]):>10}")

    elbow_diag = find_elbow(diag_k, diag_rates)
    k_elbow    = diag_k[elbow_diag]

    candidates = [
        (i, kp, ks, combined_flag_rates[i], combined_event_counts[i])
        for i, (kp, ks) in enumerate(k_pairs)
        if kp == k_elbow and ks >= k_elbow
    ]
    candidates.sort(key=lambda x: (x[4], x[3]))
    best_i, best_pt_k, best_sm_k, best_rate, best_evts = candidates[0]

    plot_elbow(diag_k, diag_rates, diag_evts, elbow_diag,
               "Combined OR signal  (diagonal: k_pt = k_sm)",
               f"{PLOT_DIR}/elbow_combined.png")

    print("=" * 60)
    print(f"\n  🏆 BEST PARAMETERS (copy to step3_detect.py):")
    print(f"     POINT_K  = {best_pt_k:.1f}")
    print(f"     SMOOTH_K = {best_sm_k:.1f}")
    print(f"     Combined flagged rate : {best_rate*100:.2f}%")
    print(f"     Combined event count  : {int(best_evts)}")
    print(f"     (elbow on diagonal at k={k_elbow:.1f}; "
          f"best pair selected by min events above elbow)")
    print("=" * 60)

if __name__ == "__main__":
    run()