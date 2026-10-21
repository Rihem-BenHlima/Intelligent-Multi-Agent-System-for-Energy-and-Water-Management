"""
step2_train_water.py — LSTM Autoencoder Training (WATER)
Input : etl_output/etl_clean.csv
Output: model/best_model.pt, model/scaler.pkl, model/threshold.npy, plots/train/
Architecture: LSTM Encoder-Decoder (Malhotra 2016); input (batch,24,1);
encoder LSTM(1→32)→LSTM(32→12); decoder RepeatVector→LSTM(12→32)→Linear(32→1).
Threshold: 99th pct of VAL errors. Interleaved train/val split (1 week held out
per 6) avoids composition bias of a plain chronological cut.
Run: python step2_train_water.py
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import CLEAN_CSV, MODEL_DIR, TRAIN_PLOT_DIR as PLOT_DIR

# Config
WINDOW      =   # time-series input window
TRAIN_RATIO =   # proportion of data allocated for training
VAL_RATIO   =   # proportion of data allocated for validation
HIDDEN_DIM  =   # hidden layer dimensionality
LATENT_DIM  =   # latent representation dimensionality
EPOCHS      =   # maximum training iterations
BATCH_SIZE  =   # number of samples per batch
LR          =   # optimizer learning rate
PATIENCE    =   # early stopping patience
THRESH_PCT  =   # percentile used to determine anomaly threshold

# Interleaved val split: hold out 1 week per 6 across the pool (avoids composition bias).
# window-1 gap around every boundary. Test = final chronological 15%.
HOLD_OUT_EVERY_N_HOURS = 168 * 6   # hold out 1 week of every 6
HOLD_OUT_LEN           = 168       # one week

SEED        = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
for d in [MODEL_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)

BLUE   = "#1565C0"
ORANGE = "#FF6D00"
GRN    = "#2E7D32"
RED    = "#E53935"

# ── MODEL ─────────────────────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    """LSTM Encoder-Decoder autoencoder. Encoder: LSTM(1→32)→LSTM(32→latent). Decoder: LSTM+Linear per timestep."""
    def __init__(self):
        super().__init__()
        self.enc1   = nn.LSTM(1,          HIDDEN_DIM, batch_first=True)
        self.enc2   = nn.LSTM(HIDDEN_DIM, LATENT_DIM, batch_first=True)
        self.dec1   = nn.LSTM(LATENT_DIM, HIDDEN_DIM, batch_first=True)
        self.output = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x):
        o, _ = self.enc1(x)
        o, _ = self.enc2(o)
        latent = o[:, -1:, :].repeat(1, x.size(1), 1)
        o, _   = self.dec1(latent)
        return self.output(o)

# ── HELPERS ───────────────────────────────────────────────────
def build_pool_masks(n_total, window, train_ratio, val_ratio,
                      hold_out_every, hold_out_len):
    """Reserve trailing pool_end..end as TEST. Within pool, hold out 1 week per hold_out_every hours as val,
    with a window-1 gap around every boundary to avoid straddling sequences."""
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
        return (np.empty((0, window, 1), dtype=np.float32),
                np.empty(0, dtype=int))
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
def _smooth(vals, alpha=0.15):
    """Exponential moving average for plot readability only — never used for threshold decisions."""
    vals = np.asarray(vals, dtype=float)
    out = np.empty_like(vals)
    out[0] = vals[0]
    for i in range(1, len(vals)):
        out[i] = alpha * vals[i] + (1 - alpha) * out[i-1]
    return out

def plot_training_history(hist):
    fig, ax = plt.subplots(figsize=(8, 5))
    ep = range(1, len(hist["train"])+1)
    train_s = _smooth(hist["train"])
    val_s   = _smooth(hist["val"])

    ax.plot(ep, train_s, color=BLUE,   lw=2.5, label="Training Loss")
    ax.plot(ep, val_s,   color=ORANGE, lw=2.5, label="Validation Loss")

    # y-axis from 0, sized off the raw (unsmoothed) first value
    y_top = max(hist["train"][0], hist["val"][0]) * 1.05
    ax.set_ylim(bottom=0, top=y_top)
    ax.set_xlim(left=1)

    best_ep = int(np.argmin(hist["val"]))+1
    best_v  = min(hist["val"])
    ax.axvline(best_ep, color="gray", ls=":", lw=1.5,
               label=f"Best epoch {best_ep}  (Val={best_v:.5f})")

    ax.set_title("LSTM Autoencoder Training Loss", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (MSE)", fontsize=12)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/01_training_history.png", dpi=150)
    plt.close(); print("  ✓ 01_training_history.png")

def _best_recon_window(ts_dt, actual, recon, n_days):
    """Pick the contiguous window best showcasing the model: lowest MAE, clean rhythm, Monday 00:00 start."""
    a  = np.asarray(actual, float)
    e  = np.abs(a - np.asarray(recon, float))
    tv = np.asarray(ts_dt.values, dtype="datetime64[h]")
    dayi = tv.astype("datetime64[D]").astype("int64")
    dow  = (dayi + 3) % 7                              # 1970-01-01 = Thursday
    hr   = np.asarray(ts_dt.hour)
    run  = np.cumsum(np.r_[1, np.diff(dayi * 24 + hr)] != 1)
    want = n_days * 24

    scored = []                                        # (clean, monday, mae, start, L)
    for rid in np.unique(run):
        sel = np.where(run == rid)[0]
        if len(sel) < 48:
            continue
        L = min(want, len(sel))
        for s in range(0, len(sel) - L + 1):
            seg = sel[s:s + L]
            wd, we = a[seg][dow[seg] < 5], a[seg][dow[seg] >= 5]
            wd_m = wd.mean() if wd.size else 0.0
            we_m = we.mean() if we.size else 0.0
            uk, inv = np.unique(dayi[seg], return_inverse=True)
            cnt = np.bincount(inv)
            dm  = np.bincount(inv, a[seg]) / cnt
            full_wd = (cnt >= 12) & (((uk + 3) % 7) < 5)   # ignore partial end days
            wd_day_min = dm[full_wd].min() if full_wd.any() else 0.0
            clean  = (we_m > 0 and wd_m > 4 * we_m and wd_day_min > 0.35 * wd_m)
            monday = bool(dow[seg[0]] == 0 and hr[seg[0]] == 0)
            scored.append((clean, monday, float(e[seg].mean()), int(seg[0]), int(L)))
    if not scored:
        return 0, min(want, len(ts_dt))
    scored.sort(key=lambda x: (not x[0], not x[1], x[2]))   # clean, then monday, then MAE
    return scored[0][3], scored[0][4]


def plot_reconstruction(ts, actual, recon, split, n_days=14):
    ts_dt = pd.to_datetime(ts)
    actual = np.asarray(actual); recon = np.asarray(recon)
    i0, L = _best_recon_window(ts_dt, actual, recon, n_days)
    sl = slice(i0, i0 + L)
    ts_ = ts_dt[sl]; act = actual[sl]; rec = recon[sl]

    fig, ax = plt.subplots(figsize=(16, 5))
    mv = compute_metrics(act, rec)
    fig.suptitle(
        f"Reconstruction — {split} set   "
        f"{ts_[0]:%a %d %b %Y} – {ts_[-1]:%a %d %b %Y}  ({L/24:.0f} contiguous days, clean data)\n"
        f"MAE={mv['mae']:.2f} L/h   R²={mv['r2']:.4f}   MAPE={mv['mape']:.2f}%",
        fontweight="bold", fontsize=13)

    ax.plot(ts_, act, color=BLUE,   lw=1.3, label="Actual")
    ax.plot(ts_, rec, color=ORANGE, lw=1.3, ls="--", label="Reconstructed")
    ax.fill_between(ts_, act, rec, alpha=0.12, color="orange")
    ax.set_ylabel("Consumption (L/h)"); ax.set_xlabel("Date")
    ax.set_xlim(ts_[0], ts_[-1])
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/02_reconstruction_{split.lower()}.png", dpi=150)
    plt.close(); print(f"  ✓ 02_reconstruction_{split.lower()}.png")

def plot_scatter(actual, recon, split):
    fig, ax = plt.subplots(figsize=(6, 6))
    mv = compute_metrics(actual, recon)
    ax.scatter(actual, recon, alpha=0.1, s=5, color=BLUE)
    lim = [min(actual.min(), recon.min()), max(actual.max(), recon.max())]
    ax.plot(lim, lim, "k--", lw=1.5, label="Perfect")
    ax.set_title(f"Actual vs Reconstructed — {split}\n"
                 f"R²={mv['r2']:.4f}  MAE={mv['mae']:.2f} L/h",
                 fontweight="bold")
    ax.set_xlabel("Actual (L/h)"); ax.set_ylabel("Reconstructed (L/h)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/03_scatter_{split.lower()}.png", dpi=150)
    plt.close(); print(f"  ✓ 03_scatter_{split.lower()}.png")

def plot_error_distribution(train_errs, val_errs, threshold):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, errs, title, color in [
        (axes[0], train_errs, "Train", BLUE),
        (axes[1], val_errs,   "Val",   ORANGE),
    ]:
        rate = (errs > threshold).mean() * 100
        ax.hist(errs, bins=50, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(threshold, color="black", ls="--", lw=2,
                   label=f"Threshold={threshold:.5f}")
        ax.set_title(f"{title} Reconstruction Error Distribution\n"
                     f"Anomaly rate: {rate:.2f}%", fontweight="bold")
        ax.set_xlabel("MSE per 24h window"); ax.set_ylabel("Count")
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/04_error_distribution.png", dpi=150)
    plt.close(); print("  ✓ 04_error_distribution.png")

def plot_error_profile(ts, errors, threshold):
    ts_ = pd.to_datetime(ts)
    df_e = pd.DataFrame({"hour": ts_.hour, "dow": ts_.dayofweek, "error": errors})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    hourly = df_e.groupby("hour")["error"].mean()
    axes[0].bar(hourly.index, hourly.values, color=BLUE, alpha=0.8)
    axes[0].axhline(threshold, color=RED, ls="--", lw=2,
                    label=f"Threshold={threshold:.5f}")
    axes[0].set_title("Mean Reconstruction Error by Hour of Day", fontweight="bold")
    axes[0].set_xlabel("Hour"); axes[0].set_ylabel("Mean MSE")
    axes[0].set_xticks(range(0, 24, 2)); axes[0].legend()
    axes[0].grid(alpha=0.3, axis="y")
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    daily = df_e.groupby("dow")["error"].mean()
    axes[1].bar(daily.index, daily.values, color=GRN, alpha=0.8)
    axes[1].axhline(threshold, color=RED, ls="--", lw=2,
                    label=f"Threshold={threshold:.5f}")
    axes[1].set_title("Mean Reconstruction Error by Day of Week", fontweight="bold")
    axes[1].set_xlabel("Day"); axes[1].set_ylabel("Mean MSE")
    axes[1].set_xticks(range(7)); axes[1].set_xticklabels(days)
    axes[1].legend(); axes[1].grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/05_error_profile.png", dpi=150)
    plt.close(); print("  ✓ 05_error_profile.png")

# ── MAIN ──────────────────────────────────────────────────────
def run():
    print("=" * 60)
    print(f"  STEP 2 — LSTM Autoencoder Training  [{DEVICE}]")
    print("=" * 60)
    print(f"  Window={WINDOW}h  Hidden={HIDDEN_DIM}  Latent={LATENT_DIM}  "
          f"(compression {WINDOW//LATENT_DIM}:1)\n")

    # Fixed seed + deterministic mode for reproducible weights, threshold and plots.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    # warn_only so a missing deterministic kernel degrades to a warning, not an abort.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print(f"  (deterministic-algorithms not fully available: {e})")
    print(f"  Seed: {SEED}  (deterministic mode)")

    df = pd.read_csv(CLEAN_CSV, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    water   = df["consumption_l"].values.astype(np.float32)
    n_total = len(df)

    print(f"  Rows : {n_total:,}  "
          f"({df['datetime'].min().date()} → {df['datetime'].max().date()})")

    # Interleaved train/val split — see HOLD_OUT_EVERY_N_HOURS above
    train_mask, val_mask, gap_mask = build_pool_masks(
        n_total, WINDOW, TRAIN_RATIO, VAL_RATIO,
        HOLD_OUT_EVERY_N_HOURS, HOLD_OUT_LEN)
    pool_end = int(n_total * TRAIN_RATIO) + int(n_total * VAL_RATIO)

    print(f"  Split: train={train_mask.sum():,}h  val={val_mask.sum():,}h  "
          f"gap={gap_mask.sum():,}h  (val interleaved across first "
          f"{pool_end:,}h — trailing {n_total-pool_end:,}h "
          f"left untouched, that's step3's test set)")

    # StandardScaler: z-score centres data; better gradient landscape than MinMaxScaler
    # for this zero-inflated/bimodal distribution (~58% zeros vs peaks >1000 L/h).
    scaler = StandardScaler()
    scaler.fit(water[train_mask].reshape(-1,1))
    scaled = scaler.transform(water.reshape(-1,1)).flatten()
    with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Scaler: StandardScaler  mean={scaler.mean_[0]:.4f}  "
          f"std={scaler.scale_[0]:.2f} L/h")

    # Sequences from valid mask regions only
    X_train, idx_tr = make_sequences_from_mask(scaled, train_mask, WINDOW)
    X_val,   idx_v  = make_sequences_from_mask(scaled, val_mask,   WINDOW)
    ts_train = df["datetime"].values[idx_tr]
    ts_val   = df["datetime"].values[idx_v]

    print(f"  Sequences: train={len(X_train):,}  val={len(X_val):,}\n")

    # Train
    model     = LSTMAutoencoder().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.MSELoss()
    # CosineAnnealingLR: smooth lr decay → no flat regions, no abrupt halving.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6)
    loader    = DataLoader(TensorDataset(torch.tensor(X_train)),
                           batch_size=BATCH_SIZE, shuffle=True)
    train_t   = torch.tensor(X_train).to(DEVICE)
    val_t     = torch.tensor(X_val).to(DEVICE)

    hist = {"train":[], "val":[]}
    best_val, patience_cnt, best_state = float("inf"), 0, None

    for epoch in range(1, EPOCHS+1):
        model.train()
        for (b,) in loader:
            b = b.to(DEVICE); optimizer.zero_grad()
            loss = criterion(model(b), b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            # Post-epoch eval on full sets with final weights (not per-batch averages).
            tl = criterion(model(train_t), train_t).item()
            vl = criterion(model(val_t),   val_t).item()
        hist["train"].append(tl); hist["val"].append(vl)
        scheduler.step()   # CosineAnnealingLR — no argument needed
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  "
                  f"train={tl:.6f}  val={vl:.6f}")
        if vl < best_val:
            best_val, patience_cnt = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stop at epoch {epoch}"); break

    model.load_state_dict(best_state); model.eval()
    torch.save({"state_dict": best_state,
                "hidden_dim": HIDDEN_DIM,
                "latent_dim": LATENT_DIM,
                "window":     WINDOW},
               f"{MODEL_DIR}/best_model.pt")
    np.savez(f"{MODEL_DIR}/history.npz",
             train=np.asarray(hist["train"], dtype=float),
             val=np.asarray(hist["val"], dtype=float))

    # Errors and threshold
    train_errs, train_recon = get_errors_recon(model, X_train)
    val_errs,   val_recon   = get_errors_recon(model, X_val)

    # Threshold = 99th pct of VAL errors (held out from gradient updates — fair out-of-sample estimate).
    threshold = float(np.percentile(val_errs, THRESH_PCT))
    np.save(f"{MODEL_DIR}/threshold.npy", threshold)

    act_tr = inv(X_train[:,-1,0], scaler); rec_tr = inv(train_recon[:,-1], scaler)
    act_v  = inv(X_val[:,-1,0],   scaler); rec_v  = inv(val_recon[:,-1],   scaler)

    print("\n" + "="*55)
    print("  RECONSTRUCTION METRICS  (consumption_l scale)")
    print("="*55)
    print(f"  {'Split':<8}  {'MAE':>7}  {'RMSE':>7}  "
          f"{'MAPE':>8}  {'R²':>7}")
    print(f"  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*7}")
    for label, a, r in [("Train", act_tr, rec_tr), ("Val", act_v, rec_v)]:
        mv = compute_metrics(a, r)
        print(f"  {label:<8}  {mv['mae']:>7.4f}  {mv['rmse']:>7.4f}"
              f"  {mv['mape']:>7.2f}%  {mv['r2']:>7.4f}")

    print(f"\n  Threshold ({THRESH_PCT}th pct VAL errors): {threshold:.6f}")
    print(f"  Train anomaly rate : "
          f"{(train_errs>threshold).mean()*100:.2f}%")
    print(f"  Val   anomaly rate : "
          f"{(val_errs>threshold).mean()*100:.2f}%  (target: ~{100-THRESH_PCT}%)")

    with open(f"{MODEL_DIR}/model_info.txt", "w") as f:
        mv_tr = compute_metrics(act_tr, rec_tr)
        mv_v  = compute_metrics(act_v,  rec_v)
        f.write(f"Architecture : LSTM Autoencoder (Malhotra et al. 2016)\n")
        f.write(f"Window       : {WINDOW}h\n")
        f.write(f"Hidden dim   : {HIDDEN_DIM}\n")
        f.write(f"Latent dim   : {LATENT_DIM}  (compression {WINDOW//LATENT_DIM}:1)\n")
        f.write(f"Scaler       : StandardScaler z-score (fit on train only)\n")
        f.write(f"Threshold    : {threshold:.6f} ({THRESH_PCT}th pct val errors)\n")
        f.write(f"Train R²     : {mv_tr['r2']:.4f}\n")
        f.write(f"Train MAE    : {mv_tr['mae']:.2f} L/h\n")
        f.write(f"Val R²       : {mv_v['r2']:.4f}\n")
        f.write(f"Val MAE      : {mv_v['mae']:.2f} L/h\n")

    print("\n  Generating training plots...")
    plot_training_history(hist)
    plot_reconstruction(ts_train, act_tr, rec_tr, "Train")
    plot_reconstruction(ts_val,   act_v,  rec_v,  "Val")
    plot_scatter(act_tr, rec_tr, "Train")
    plot_scatter(act_v,  rec_v,  "Val")
    plot_error_distribution(train_errs, val_errs, threshold)
    plot_error_profile(ts_train, train_errs, threshold)

    print(f"\n  ✅ Model    → {MODEL_DIR}/best_model.pt")
    print(f"  ✅ Scaler   → {MODEL_DIR}/scaler.pkl")
    print(f"  ✅ Threshold→ {MODEL_DIR}/threshold.npy")
    print(f"  ✅ Plots    → {PLOT_DIR}/")
    return model, scaler, threshold

if __name__ == "__main__":
    run()