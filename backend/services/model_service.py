"""
Singleton ModelStore registry that loads ML/DL forecasting and anomaly detection models.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Path resolution ───────────────────────────────────────────────────────────
_env = os.getenv("ML_MODELS_DIR")
_ROOT: Path = Path(_env) if _env else Path(__file__).resolve().parent.parent.parent

# Energy forecasting
_E_ML   = _ROOT / "energy_forecasting" / "training" / "ml"
_E_RF   = _E_ML / "randomforest" / "output" / "rf_production.joblib"
_E_XGB  = _E_ML / "xgboost" / "output" / "xgboost_production.joblib"

# Water forecasting — single-step (t+1) models, joblib-saved
_W_ML   = _ROOT / "water_forecasting" / "training" / "ml"
_W_RF   = _W_ML / "random_forest" / "model" / "rf_model.joblib"
_W_LGB  = _W_ML / "lightgbm"      / "model" / "lgb_model.joblib"

# Anomaly detection
_E_ANOM = _ROOT / "energy_anomaly_detection" / "model"
_W_ANOM = _ROOT / "water_anomaly_detection"  / "model"

# Window length (days) for rolling threshold calibration
ANOMALY_DETECTION_DAYS: int = 900


# ── LSTM Autoencoder architecture (must match training) ───────────────────────
class LSTMAutoencoder(nn.Module):
    """
    LSTM Encoder-Decoder (Malhotra et al. 2016).
    Input : (batch, 24, 1)
    Encoder: LSTM(1→32) → LSTM(32→12)
    Decoder: repeat latent → LSTM(12→32) → Linear(32→1)
    """
    HIDDEN_DIM = 32
    LATENT_DIM = 12

    def __init__(self):
        super().__init__()
        H, L = self.HIDDEN_DIM, self.LATENT_DIM
        self.enc1   = nn.LSTM(1, H, batch_first=True)
        self.enc2   = nn.LSTM(H, L, batch_first=True)
        self.dec1   = nn.LSTM(L, H, batch_first=True)
        self.output = nn.Linear(H, 1)

    def forward(self, x):
        o, _ = self.enc1(x)
        o, _ = self.enc2(o)
        latent = o[:, -1:, :].repeat(1, x.size(1), 1)
        o, _  = self.dec1(latent)
        return self.output(o)


def _load_autoencoder(model_path: Path) -> Optional[nn.Module]:
    """Load a saved LSTMAutoencoder checkpoint (state_dict dict or full module)."""
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model = LSTMAutoencoder()
        # The training script saves {"state_dict": ..., "hidden_dim": ..., ...}
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            # Fallback: checkpoint is the raw state_dict or full module
            try:
                model.load_state_dict(checkpoint)
            except Exception:
                model = checkpoint  # already a module
        model.eval()
        logger.info("Loaded autoencoder from %s", model_path)
        return model
    except FileNotFoundError:
        logger.warning("Autoencoder not found at %s", model_path)
    except Exception as e:
        logger.warning("Failed to load autoencoder %s: %s", model_path, e)
    return None


# ── ModelStore ────────────────────────────────────────────────────────────────
class ModelStore:
    """Class-level model registry. Call ``ModelStore.load()`` once at startup."""

    # Energy forecasting
    energy_rf = None   # RandomForest (Best from Ensemble CV R²=0.79)

    # Water forecasting
    water_rf         = None   # RandomForest NSE=0.793 (primary)
    water_lgb        = None   # LightGBM     NSE=0.790 (backup)

    # Anomaly detection — energy
    energy_anom_model     = None
    energy_anom_scaler    = None
    energy_anom_threshold: Optional[float] = None
    energy_anom_config: Optional[dict] = None       # step4_export_detection_config.py output
    energy_anom_val_seed: Optional[np.ndarray] = None
    energy_anom_val_smooth_seed: Optional[np.ndarray] = None

    # Anomaly detection — water
    water_anom_model      = None
    water_anom_scaler     = None
    water_anom_threshold:  Optional[float] = None
    water_anom_config: Optional[dict] = None        # step4_export_detection_config.py output

    # ── load ──────────────────────────────────────────────────────────────────
    @classmethod
    def load(cls) -> None:
        """Load all model artefacts. Missing files emit warnings but never raise."""

        def _joblib(path: Path, label: str):
            try:
                m = joblib.load(path)
                logger.info("Loaded %-30s from %s", label, path)
                return m
            except FileNotFoundError:
                logger.warning("%-30s NOT FOUND at %s", label, path)
            except Exception as e:
                logger.warning("%-30s failed to load: %s", label, e)
            return None

        # Energy forecasting
        cls.energy_rf = _joblib(_E_RF, "Energy RandomForest (primary)")
        cls.energy_xgb = _joblib(_E_XGB, "Energy XGBoost (backup)")

        # Water forecasting
        cls.water_rf  = _joblib(_W_RF, "Water RandomForest (primary)")
        cls.water_lgb = _joblib(_W_LGB, "Water LightGBM (backup)")

        # Energy anomaly detection
        cls.energy_anom_model  = _load_autoencoder(_E_ANOM / "best_model.pt")
        cls.energy_anom_scaler = _joblib(_E_ANOM / "scaler.pkl", "Energy anomaly scaler")
        try:
            cls.energy_anom_threshold = float(np.load(_E_ANOM / "threshold.npy"))
            logger.info("Energy anomaly threshold: %.6f", cls.energy_anom_threshold)
        except Exception as e:
            logger.warning("Energy anomaly threshold failed: %s", e)
        try:
            with open(_E_ANOM / "detection_config.json") as f:
                cls.energy_anom_config = json.load(f)
            seeds = np.load(_E_ANOM / "val_seed.npz")
            cls.energy_anom_val_seed = seeds["val_seed"]
            cls.energy_anom_val_smooth_seed = seeds["val_smooth_seed"]
            logger.info("Loaded Energy anomaly detection config (rolling, point_floor=%.6f)",
                        cls.energy_anom_config["point_floor"])
        except Exception as e:
            logger.warning("Energy anomaly detection_config failed to load: %s", e)

        # Water anomaly detection
        cls.water_anom_model  = _load_autoencoder(_W_ANOM / "best_model.pt")
        cls.water_anom_scaler = _joblib(_W_ANOM / "scaler.pkl", "Water anomaly scaler")
        try:
            cls.water_anom_threshold = float(np.load(_W_ANOM / "threshold.npy"))
            logger.info("Water anomaly threshold: %.6f", cls.water_anom_threshold)
        except Exception as e:
            logger.warning("Water anomaly threshold failed: %s", e)
        try:
            with open(_W_ANOM / "detection_config.json") as f:
                cls.water_anom_config = json.load(f)
            logger.info("Loaded Water anomaly detection config (fixed, point_threshold=%.6f)",
                        cls.water_anom_config["point_threshold"])
        except Exception as e:
            logger.warning("Water anomaly detection_config failed to load: %s", e)

    # Feature set matching energy model training contract and final_feature_set.txt order.
    ENERGY_FEATURES = [
        "lag_144h", "is_work", "lag_120h", "lag_168h", "lag_336h", "lag_1h", "lag_72h", "lag_48h",
        "we_x_lag168", "work_x_month", "profile_median", "solar_rad_24h", "profile_median_summer",
        "t_dow", "dow_sin", "lag_24h", "lag_96h", "hour_sin", "load_velocity", "roll_max_24h",
        "is_weekend", "roll_mean_3h", "solar_rad_obs", "hour_cos2", "roll_mean_168h", "roll_std_168h",
        "dow_x_hour", "lag_73h", "lag_49h", "roll_max_48h", "dew_point_obs", "night_min_24h",
        "roll_std_6h", "roll_std_3h", "roll_min_24h", "roll_std_72h", "is_afternoon", "summer_x_lag168",
        "doy_cos", "summer_x_lag24", "temp_max_24h", "roll_std_12h", "lag_25h", "roll_mean_24h",
        "is_night", "work_x_cdd24", "hour_sin2", "lag_3h", "is_morning_peak", "day_night_delta",
        "is_midday", "is_holiday",
    ]

    # ── predict_energy ────────────────────────────────────────────────────────
    @classmethod
    def predict_energy(cls, features_df: pd.DataFrame) -> dict:
        """Run energy forecast using primary RandomForest with XGBoost fallback."""
        # Ensure correct column order
        for col in cls.ENERGY_FEATURES:
            if col not in features_df.columns:
                features_df[col] = 0.0
        features_df = features_df[cls.ENERGY_FEATURES]

        # Try primary model (RandomForest)
        if cls.energy_rf is not None:
            try:
                preds = cls.energy_rf.predict(features_df)
                return {"predictions": preds.tolist(), "model_used": "RandomForest", "status": "success"}
            except Exception as e:
                logger.warning(f"RandomForest failed: {e}. Falling back to XGBoost.")

        # Try backup model (XGBoost)
        if cls.energy_xgb is not None:
            try:
                preds = cls.energy_xgb.predict(features_df)
                return {"predictions": preds.tolist(), "model_used": "XGBoost", "status": "fallback"}
            except Exception as e:
                logger.error(f"XGBoost backup failed: {e}")

        return {"predictions": [], "model_used": "none", "status": "error"}

    # Feature ordering matching the water model feature set contract.
    WATER_FEATURES = [
        'door_open_prob', 'lag_168h', 'hour', 'lag_1h', 'rolling_mean_3h',
        'is_working_hour', 'hour_cos', 'ewma_6h', 'lag_2h', 'lag_24h',
        'rolling_mean_6h', 'diff_1h', 'diff_24h', 'hour_sin', 'lag_3h',
    ]

    # ── predict_water ─────────────────────────────────────────────────────────
    @classmethod
    def predict_water(cls, features_df: pd.DataFrame) -> dict:
        """
        Run a single-step (t+1) water forecast, with fallback RandomForest → LightGBM.
        Caller is responsible for recursive multi-step forecasting (feed each
        prediction back into the next step's lag/rolling/ewma features) — see
        the water branch of /api/forecast/{domain} in main.py.
        """
        # Ensure correct column order
        for col in cls.WATER_FEATURES:
            if col not in features_df.columns:
                features_df[col] = 0.0
        features_df = features_df[cls.WATER_FEATURES]

        chain = [
            (cls.water_rf, "random_forest", "ok"),
            (cls.water_lgb, "lightgbm", "fallback"),
        ]
        for model, name, status in chain:
            if model is None:
                continue
            try:
                preds = np.asarray(model.predict(features_df))
                return {"predictions": preds.tolist(), "model_used": name, "status": status}
            except Exception as e:
                logger.warning("Water forecast failed with %s: %s", name, e)

        logger.error("All water forecast models failed.")
        return {"predictions": [], "model_used": "none", "status": "error"}

    # ── detect_anomalies (shared logic) ───────────────────────────────────────
    # Autoencoder sliding window reconstruction error detection.
    # Energy uses a rolling threshold; Water uses a fixed threshold.

    @staticmethod
    def _make_sequences(values: np.ndarray, window: int):
        """Build sliding window sequences."""
        X, idx = [], []
        for i in range(len(values) - window + 1):
            X.append(values[i:i + window])
            idx.append(i + window - 1)
        if not X:
            return np.empty((0, window, 1), dtype=np.float32), np.array([], dtype=int)
        return np.array(X, dtype=np.float32)[:, :, np.newaxis], np.array(idx)

    @staticmethod
    def _get_window_errors(model: nn.Module, X: np.ndarray):
        """Compute per-window MSE and reconstructed value at the window's last timestep."""
        with torch.no_grad():
            t = torch.tensor(X)
            r = model(t)
            errs = ((r - t) ** 2).mean(dim=[1, 2]).cpu().numpy()
            recon_last = r[:, -1, 0].cpu().numpy()
            return errs, recon_last

    @staticmethod
    def _rolling_threshold(errors: np.ndarray, k: float, window: int, lag: int,
                            floor: float, seed: Optional[np.ndarray]) -> np.ndarray:
        """Compute rolling mean + k * std threshold with 95th percentile clipping and warm-start seed."""
        errors = np.asarray(errors, dtype=np.float64)
        if seed is not None and len(seed) > 0:
            seed = np.asarray(seed, dtype=np.float64)
            n_seed = len(seed)
            errors_ext = np.concatenate([seed, errors])
        else:
            n_seed = 0
            errors_ext = errors

        clip_val = np.percentile(errors_ext[n_seed:], 95) if len(errors_ext[n_seed:]) else 0.0
        eb = np.clip(errors_ext, 0, clip_val)

        s  = pd.Series(eb)
        rm = s.rolling(window, min_periods=1).mean()
        rs = s.rolling(window, min_periods=1).std().fillna(0)
        thresh = (rm.shift(lag).fillna(rm.iloc[0]) +
                  k * rs.shift(lag).fillna(rs.iloc[0])).values

        thresh = thresh[n_seed:]
        return np.maximum(thresh, floor)

    @staticmethod
    def _group_events(timestamps: list, mask: np.ndarray, gap_h: int):
        """Group consecutive flagged timestamps within gap_h hours into discrete events."""
        ts = pd.to_datetime(pd.Series(timestamps)[mask])
        if len(ts) == 0:
            return []
        ts = ts.tolist()
        events, s, e = [], ts[0], ts[0]
        for t in ts[1:]:
            if (t - e).total_seconds() / 3600 <= gap_h:
                e = t
            else:
                events.append((s, e))
                s = e = t
        events.append((s, e))
        return events

    @classmethod
    def _run_anomaly_detection(
        cls,
        consumption_series: list,
        timestamps: list,
        model: Optional[nn.Module],
        scaler,
        config: Optional[dict],
        val_seed: Optional[np.ndarray],
        val_smooth_seed: Optional[np.ndarray],
        domain: str,
    ) -> dict:
        """Run sliding window reconstruction error detection and event grouping."""
        empty = {"events": [], "n_windows": 0, "method": None, "timeline": []}

        if model is None or scaler is None or config is None:
            logger.warning("%s anomaly model/scaler/config not loaded — returning empty results.", domain)
            return empty

        window = config["window"]
        if len(consumption_series) < window:
            logger.warning("%s anomaly detection: series too short (%d < %d window)",
                            domain, len(consumption_series), window)
            return empty

        try:
            values = np.array(consumption_series, dtype=np.float32)
            scaled = scaler.transform(values.reshape(-1, 1)).flatten()
            X, idx = cls._make_sequences(scaled, window)
            if len(X) == 0:
                return empty

            point_errs, recon_scaled_last = cls._get_window_errors(model, X)
            smooth_errs = pd.Series(point_errs).rolling(
                config["smooth_win"], min_periods=1).mean().values

            # Reconstruct original units for actual-vs-expected explainability
            actual_orig = values[idx]
            expected_orig = scaler.inverse_transform(
                recon_scaled_last.reshape(-1, 1)).flatten()

            if config["method"] == "rolling":
                point_thresh = cls._rolling_threshold(
                    point_errs, config["point_k"], config["thresh_win"],
                    config["thresh_lag"], config["point_floor"], val_seed)
                smooth_thresh = cls._rolling_threshold(
                    smooth_errs, config["smooth_k"], config["thresh_win"],
                    config["thresh_lag"], config["smooth_floor"], val_smooth_seed)
                point_floor = config["point_floor"]
                smooth_floor = config["smooth_floor"]
            else:  # "fixed"
                point_thresh = np.full(len(point_errs), config["point_threshold"])
                smooth_thresh = np.full(len(point_errs), config["smooth_threshold"])
                point_floor = config["point_threshold"]
                smooth_floor = config["smooth_threshold"]

            mask_point = point_errs > point_thresh
            mask_smooth = smooth_errs > smooth_thresh
            mask = mask_point | mask_smooth

            ts_windows = [timestamps[i] for i in idx]
            raw_events = cls._group_events(ts_windows, mask, config["gap_h"])

            ts_series = pd.to_datetime(pd.Series(ts_windows))
            events = []
            for start, end in raw_events:
                in_ev = (ts_series >= start) & (ts_series <= end)
                dur_h = int((end - start).total_seconds() / 3600) + 1
                if dur_h < config["min_event_h"]:
                    continue
                max_e = float(point_errs[in_ev.values].max())
                max_e_sm = float(smooth_errs[in_ev.values].max())
                fires_pt = bool(mask_point[in_ev.values].any())
                fires_sm = bool(mask_smooth[in_ev.values].any())
                signal = "BOTH" if (fires_pt and fires_sm) else ("PT" if fires_pt else "SM")
                margin_pt = (max_e - point_floor) / point_floor if point_floor else 0.0
                margin_sm = (max_e_sm - smooth_floor) / smooth_floor if smooth_floor else 0.0
                margin = min(margin_pt, margin_sm) if signal == "BOTH" else (
                    margin_pt if signal == "PT" else margin_sm)

                # Hour within this event with peak reconstruction error
                in_ev_idx = np.where(in_ev.values)[0]
                peak_idx = in_ev_idx[np.argmax(point_errs[in_ev_idx])]

                events.append({
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "duration_h": dur_h,
                    "max_mse": max_e,
                    "signal": signal,
                    "margin": round(margin, 4),
                    "peak_time": pd.Timestamp(ts_windows[peak_idx]).isoformat(),
                    "peak_actual": round(float(actual_orig[peak_idx]), 3),
                    "peak_expected": round(max(0.0, float(expected_orig[peak_idx])), 3),
                })

            # Rebuild mask from filtered events meeting min_event_h
            event_mask = np.zeros(len(point_errs), dtype=bool)
            for start, end in raw_events:
                dur_h = int((end - start).total_seconds() / 3600) + 1
                if dur_h < config["min_event_h"]:
                    continue
                in_ev = (ts_series >= start) & (ts_series <= end)
                event_mask |= in_ev.values

            # Timeline of reconstruction errors and thresholds
            timeline = [
                {
                    "timestamp": ts_windows[i].isoformat() if hasattr(ts_windows[i], "isoformat")
                                 else pd.Timestamp(ts_windows[i]).isoformat(),
                    "point_error": float(point_errs[i]),
                    "point_threshold": float(point_thresh[i]),
                    "smooth_error": float(smooth_errs[i]),
                    "smooth_threshold": float(smooth_thresh[i]),
                    "flagged": bool(event_mask[i]),
                }
                for i in range(len(point_errs))
            ]

            logger.info("%s anomaly detection: %d windows, %d event(s)",
                        domain, len(X), len(events))
            return {"events": events, "n_windows": len(X), "method": config["method"],
                    "timeline": timeline}
        except Exception as e:
            logger.error("%s anomaly detection failed: %s", domain, e, exc_info=True)
            return empty

    @classmethod
    def detect_energy_anomalies(cls, consumption_series: list, timestamps: list) -> dict:
        """Detect anomalies in energy consumption time series."""
        return cls._run_anomaly_detection(
            consumption_series, timestamps,
            cls.energy_anom_model, cls.energy_anom_scaler, cls.energy_anom_config,
            cls.energy_anom_val_seed, cls.energy_anom_val_smooth_seed,
            "Energy",
        )

    @classmethod
    def detect_water_anomalies(cls, consumption_series: list, timestamps: list) -> dict:
        """Detect anomalies in water consumption time series."""
        result = cls._run_anomaly_detection(
            consumption_series, timestamps,
            cls.water_anom_model, cls.water_anom_scaler, cls.water_anom_config,
            None, None,
            "Water",
        )
        cls._snap_water_events_to_days(result)
        return result

    @staticmethod
    def _snap_water_events_to_days(result: dict) -> None:
        """Align water anomaly event start/end boundaries to calendar days (00:00 to 23:59:59)."""
        if not result["events"]:
            return
        for ev in result["events"]:
            start = pd.Timestamp(ev["start"]).normalize()
            end = pd.Timestamp(ev["end"]).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            ev["start"] = start.isoformat()
            ev["end"] = end.isoformat()
            ev["duration_h"] = int((end - start).total_seconds() / 3600) + 1
            for point in result["timeline"]:
                pt_ts = pd.Timestamp(point["timestamp"])
                if start <= pt_ts <= end:
                    point["flagged"] = True
