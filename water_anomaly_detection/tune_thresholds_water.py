"""
tune_thresholds_water.py — Threshold Selection via Elbow Method (WATER, FIXED)
Selects POINT_K and SMOOTH_K for a fixed threshold (mean + k·std of train errors).
For each k, computes flagged rate and event count on test; optimal k = elbow of flagged-rate curve.
Output: plots/tune/elbow_combined.png
Run: python tune_thresholds_water.py
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

from config import CLEAN_CSV, RAW_CSV, MODEL_DIR, TUNE_PLOT_DIR as PLOT_DIR

# ── CONFIG (must match step2_train_water.py and step3_detect_water.py) ────
WINDOW      = 24
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
HOLD_OUT_EVERY_N_HOURS = 168 * 6
HOLD_OUT_LEN           = 168
HIDDEN_DIM  =     # hidden layer dimensionality
LATENT_DIM  =     # latent representation dimensionality
SMOOTH_WIN  =     # smoothing window size
GAP_H       = 6       # hours gap to merge consecutive flags into one event
MIN_EVENT_H = 2        # minimum event duration to count (drops noise)
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
def make_sequences(values, window):
    X, idx = [], []
    for i in range(len(values) - window + 1):
        X.append(values[i:i+window]); idx.append(i+window-1)
    return np.array(X, dtype=np.float32)[:,:,np.newaxis], np.array(idx)

def build_pool_masks(n_total, window, train_ratio, val_ratio,
                      hold_out_every, hold_out_len):
    """Same scheme as step2_train_water.py: reserve trailing pool_end..end as TEST;
    hold out 1 week per hold_out_every as val, with a window-1 gap at boundaries."""
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
    return np.array(X, dtype=np.float32)[:, :, np.newaxis], np.array(idx)

def get_errors(model, X):
    loader = DataLoader(TensorDataset(torch.tensor(X)), batch_size=512, shuffle=False)
    errs = []
    with torch.no_grad():
        for (b,) in loader:
            b = b.to(DEVICE); r = model(b)
            errs.append(((r - b)**2).mean(dim=[1,2]).cpu().numpy())
    return np.concatenate(errs)

def fixed_threshold(baseline_errors, k):
    """Fixed threshold: mean + k·std of baseline errors. No rolling window."""
    baseline_errors = np.asarray(baseline_errors, dtype=np.float64)
    return float(baseline_errors.mean() + k * baseline_errors.std())

def count_events(ts, mask, gap_h=GAP_H, min_h=MIN_EVENT_H):
    """Count distinct events from a boolean mask (same logic as step3_detect_water.py)."""
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
    """Elbow via maximum perpendicular distance from the line connecting first and last curve points."""
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
    ax1.set_xlabel("k  (fixed threshold = train mean + k·σ)")
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
    print("  THRESHOLD TUNING — Elbow Method, FIXED threshold (no ground truth)")
    print("=" * 60)

    # Load model & scaler
    ckpt  = torch.load(f"{MODEL_DIR}/best_model.pt", map_location=DEVICE, weights_only=False)
    model = LSTMAutoencoder().to(DEVICE)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Load data
    df_clean = pd.read_csv(CLEAN_CSV, parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
    df_raw   = pd.read_csv(RAW_CSV,   parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)

    n_total  = len(df_clean)
    pool_end = int(n_total * TRAIN_RATIO) + int(n_total * VAL_RATIO)

    # Use saved scaler as-is — do NOT refit here.
    water_clean = df_clean["consumption_l"].values.astype(np.float32)

    # Train sequences for the fixed-threshold baseline
    train_mask, _, _ = build_pool_masks(
        n_total, WINDOW, TRAIN_RATIO, VAL_RATIO,
        HOLD_OUT_EVERY_N_HOURS, HOLD_OUT_LEN)
    scaled_clean   = scaler.transform(water_clean.reshape(-1, 1)).flatten()
    X_train, _     = make_sequences_from_mask(scaled_clean, train_mask, WINDOW)
    train_errs     = get_errors(model, X_train)
    train_smooth   = pd.Series(train_errs).rolling(SMOOTH_WIN, min_periods=1).mean().values
    print(f"  Train sequences : {len(X_train):,}  "
          f"(baseline for the fixed threshold)")

    # WINDOW-1 gap buffer at pool/test boundary — same as step3_detect_water.py
    water_raw      = df_raw["consumption_l"].values.astype(np.float32)
    scaled_raw     = scaler.transform(water_raw.reshape(-1, 1)).flatten()
    X_raw, idx_raw = make_sequences(scaled_raw, WINDOW)
    test_start_k   = pool_end
    ts_test        = pd.to_datetime(df_raw["datetime"].values[idx_raw][test_start_k:])
    X_test         = X_raw[test_start_k:]
    test_errs      = get_errors(model, X_test)
    smooth_errs    = pd.Series(test_errs).rolling(SMOOTH_WIN, min_periods=1).mean().values
    print(f"  Test sequences  : {len(X_test):,}")
    print(f"  Test range      : {ts_test[0].date()} → {ts_test[-1].date()}")
    print("-" * 60)

    # Joint grid over (k_pt, k_sm) combined OR mask; diagonal (k_pt==k_sm) used for elbow plot.
    print(f"\n  {'k_pt':>5} {'k_sm':>5} {'OR_rate':>9} {'OR_events':>10}")
    print(f"  {'-'*5} {'-'*5} {'-'*9} {'-'*10}")

    combined_flag_rates  = []
    combined_event_counts= []
    k_pairs              = []   # (k_pt, k_sm) tried

    for k_pt in K_RANGE:
        for k_sm in K_RANGE:
            t_pt = fixed_threshold(train_errs,   k_pt)
            t_sm = fixed_threshold(train_smooth, k_sm)
            m_or = (test_errs > t_pt) | (smooth_errs > t_sm)
            rate = m_or.mean()
            evts = count_events(ts_test, m_or)
            combined_flag_rates.append(rate)
            combined_event_counts.append(evts)
            k_pairs.append((k_pt, k_sm))

    combined_flag_rates   = np.array(combined_flag_rates)
    combined_event_counts = np.array(combined_event_counts, dtype=float)

    # Diagonal slice for the elbow plot
    diag_idx   = [i for i,(kp,ks) in enumerate(k_pairs) if kp == ks]
    diag_k     = np.array([k_pairs[i][0] for i in diag_idx])
    diag_rates = combined_flag_rates[diag_idx]
    diag_evts  = combined_event_counts[diag_idx]

    for i in diag_idx:
        kp, ks = k_pairs[i]
        print(f"  {kp:>5.1f} {ks:>5.1f} {combined_flag_rates[i]*100:>8.2f}% {int(combined_event_counts[i]):>10}")

    elbow_diag = find_elbow(diag_k, diag_rates)
    k_elbow    = diag_k[elbow_diag]

    # Best pair: elbow on diagonal; allow higher k_sm to suppress smooth false alarms.
    candidates = [
        (i, kp, ks, combined_flag_rates[i], combined_event_counts[i])
        for i, (kp, ks) in enumerate(k_pairs)
        if kp == k_elbow and ks >= k_elbow
    ]
    candidates.sort(key=lambda x: (x[4], x[3]))
    best_i, best_pt_k, best_sm_k, best_rate, best_evts = candidates[0]

    plot_elbow(diag_k, diag_rates, diag_evts, elbow_diag,
               "Combined OR signal, FIXED threshold  (diagonal: k_pt = k_sm)",
               f"{PLOT_DIR}/elbow_combined.png")

    print("=" * 60)
    print(f"\n  🏆 BEST PARAMETERS (copy to step3_detect_water.py):")
    print(f"     POINT_K  = {best_pt_k:.1f}")
    print(f"     SMOOTH_K = {best_sm_k:.1f}")
    print(f"     Combined flagged rate : {best_rate*100:.2f}%")
    print(f"     Combined event count  : {int(best_evts)}")
    print(f"     (elbow on diagonal at k={k_elbow:.1f}; "
          f"best pair selected by min events above elbow)")
    print("=" * 60)

if __name__ == "__main__":
    run()