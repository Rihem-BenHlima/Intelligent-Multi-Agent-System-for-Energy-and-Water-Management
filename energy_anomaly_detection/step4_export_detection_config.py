"""
step4_export_detection_config.py — Export live-serving detection config
=========================================================================
Input : model/best_model.pt, model/scaler.pkl, model/val_mask.npy,
        etl_output/etl_raw.csv
Output: model/detection_config.json

Computes the calibration constants the backend needs to replicate
step3_detect.py's dual-signal adaptive rolling-threshold methodology at
request time without re-deriving them on every call: the point/smooth
signal floors (FLOOR_PCT-th pct of RAW val errors). No warm-start seed —
step3 cold-starts the rolling window. Re-run this after every retrain of
step2_train.py, and whenever step3_detect.py's FLOOR_PCT / POINT_K /
SMOOTH_K change — the values below must stay in sync with that file.

Run:  python step4_export_detection_config.py
"""
import json
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import RAW_CSV, MODEL_DIR

WINDOW = 24
HIDDEN_DIM = 32
LATENT_DIM = 12
THRESH_WIN = 168
FLOOR_PCT = 80   # keep in sync with step3_detect.py
SMOOTH_WIN = 4


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


ckpt = torch.load(f"{MODEL_DIR}/best_model.pt", map_location="cpu", weights_only=False)
model = LSTMAutoencoder()
model.load_state_dict(ckpt["state_dict"])
model.eval()

with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

df = pd.read_csv(RAW_CSV, parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)
values = df["energy_kwh"].values.astype(np.float32)
n_total = len(values)

val_mask = np.load(f"{MODEL_DIR}/val_mask.npy")
assert len(val_mask) == n_total, "val_mask.npy length mismatch — retrain first"

scaled = scaler.transform(values.reshape(-1, 1)).flatten()
X_val, _ = make_sequences_from_mask(scaled, val_mask, WINDOW)
val_errs = get_errors(model, X_val)
val_smooth = pd.Series(val_errs).rolling(SMOOTH_WIN, min_periods=1).mean().values

point_floor = float(np.percentile(val_errs, FLOOR_PCT))
smooth_floor = float(np.percentile(val_smooth, FLOOR_PCT))

config = {
    "method": "rolling",
    "window": WINDOW,
    "smooth_win": SMOOTH_WIN,
    "thresh_win": THRESH_WIN,
    "thresh_lag": 4,
    "point_k": 3.5,
    "smooth_k": 6.0,
    "gap_h": 6,
    "min_event_h": 2,
    "low_conf_margin": 2.0,
    "point_floor": point_floor,
    "smooth_floor": smooth_floor,
    "warm_start_seed": False,   # step3 cold-starts the rolling window
}
with open(f"{MODEL_DIR}/detection_config.json", "w") as f:
    json.dump(config, f, indent=2)

print(json.dumps(config, indent=2))
print(f"Saved -> {MODEL_DIR}/detection_config.json")
