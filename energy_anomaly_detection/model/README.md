# Energy anomaly model 

## What ends up here after training

| File | Written by | Purpose |
|---|---|---|
| `best_model.pt` | `step2_train.py` | LSTM autoencoder weights (window 24 h, latent 12) |
| `scaler.pkl` | `step2_train.py` | StandardScaler fitted on the training split only |
| `threshold.npy` | `step2_train.py` | Reference anomaly threshold (95th percentile of validation error) |
| `val_mask.npy` | `step2_train.py` | Which hours belong to the validation split (reused by `step3_detect.py`) |
| `history.npy`, `history.npz` | `step2_train.py` | Training / validation loss curves |
| `model_info.txt` | `step2_train.py` | Human-readable summary (architecture, R², MAE, threshold) |
| `detection_config.json` | `step4_export_detection_config.py` | Rolling-threshold settings the backend uses at inference time |

## Who needs these files

The FastAPI backend (`backend/services/model_service.py`) loads **`best_model.pt`** and
**`scaler.pkl`** from this folder, and reads `detection_config.json` for the detection
settings. Without them the energy anomaly endpoints cannot run.