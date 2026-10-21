"""
step4_export_detection_config.py — Export live-serving detection config.
Input : model/best_model.pt, model/scaler.pkl, etl_output/etl_raw.csv
Output: model/detection_config.json
Re-run after every retrain. Run: python step4_export_detection_config.py
"""
import json
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import RAW_CSV, MODEL_DIR

WINDOW = 24
HIDDEN_DIM =   # hidden layer dimensionality
LATENT_DIM =   # latent representation dimensionality
SMOOTH_WIN =   # smoothing window size
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
HOLD_OUT_EVERY_N_HOURS = 168 * 6
HOLD_OUT_LEN = 168
POINT_K =      # point anomaly detection sensitivity
SMOOTH_K =     # smoothed anomaly detection sensitivity


class LSTMAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.LSTM(1, HIDDEN_DIM, batch_first=True)
        self.enc2 = nn.LSTM(HIDDEN_DIM, LATENT_DIM, batch_first=True)
        self.dec1 = nn.LSTM(LATENT_DIM, HIDDEN_DIM, batch_first=True)
        self.output = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x):
        o, _ = self.enc1(x)
        o, _ = self.enc2(o)
        l = o[:, -1:, :].repeat(1, x.size(1), 1)
        o, _ = self.dec1(l)
        return self.output(o)


def build_pool_masks(n_total, window, train_ratio, val_ratio, hold_out_every, hold_out_len):
    pool_end = int(n_total * train_ratio) + int(n_total * val_ratio)
    is_val = np.zeros(n_total, dtype=bool)
    pos = 0
    while pos < pool_end:
        val_start = pos + hold_out_every - hold_out_len
        val_end = pos + hold_out_every
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
    val_mask = is_val & pool_mask & (~gap_mask)
    return train_mask, val_mask, gap_mask


def make_sequences_from_mask(values, valid_mask, window):
    X, idx = [], []
    for i in range(len(values) - window + 1):
        if valid_mask[i:i + window].all():
            X.append(values[i:i + window]); idx.append(i + window - 1)
    return np.array(X, dtype=np.float32)[:, :, np.newaxis], np.array(idx)


def get_errors(model, X):
    with torch.no_grad():
        t = torch.tensor(X)
        r = model(t)
        return ((r - t) ** 2).mean(dim=[1, 2]).numpy()


def fixed_threshold(baseline_errors, k):
    baseline_errors = np.asarray(baseline_errors, dtype=np.float64)
    return float(baseline_errors.mean() + k * baseline_errors.std())


ckpt = torch.load(f"{MODEL_DIR}/best_model.pt", map_location="cpu", weights_only=False)
model = LSTMAutoencoder()
model.load_state_dict(ckpt["state_dict"])
model.eval()

with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

df = pd.read_csv(RAW_CSV, parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)
values = df["consumption_l"].values.astype(np.float32)
n_total = len(values)

train_mask, _, _ = build_pool_masks(
    n_total, WINDOW, TRAIN_RATIO, VAL_RATIO, HOLD_OUT_EVERY_N_HOURS, HOLD_OUT_LEN)

scaled = scaler.transform(values.reshape(-1, 1)).flatten()
X_train, _ = make_sequences_from_mask(scaled, train_mask, WINDOW)
train_errs = get_errors(model, X_train)
train_smooth = pd.Series(train_errs).rolling(SMOOTH_WIN, min_periods=1).mean().values

point_thresh = fixed_threshold(train_errs, POINT_K)
smooth_thresh = fixed_threshold(train_smooth, SMOOTH_K)

config = {
    "method": "fixed",
    "window": WINDOW,
    "smooth_win": SMOOTH_WIN,
    "point_k": POINT_K,
    "smooth_k": SMOOTH_K,
    "gap_h": 6,
    "min_event_h": 2,
    "low_conf_margin": 2.0,
    "point_threshold": point_thresh,
    "smooth_threshold": smooth_thresh,
}
with open(f"{MODEL_DIR}/detection_config.json", "w") as f:
    json.dump(config, f, indent=2)

print(json.dumps(config, indent=2))
print(f"Saved -> {MODEL_DIR}/detection_config.json")
