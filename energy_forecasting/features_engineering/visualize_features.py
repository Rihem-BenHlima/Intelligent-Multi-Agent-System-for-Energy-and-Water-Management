"""
EDA and correlation visualization for 24h energy forecasting.

Produces 8 publication-quality figures:
  Fig A  — Hourly energy profile (mean by hour)
  Fig B  — Full correlation matrix (features + target)
  Fig C  — Autocorrelation function (lags 0–200h)
  Fig D  — Energy vs temperature scatter (CDD hockey stick)
  Fig E  — Weather variable correlations with target
  Fig F  — Seasonal energy profiles (by month)
  Fig G  — Work vs non-work day profiles
  Fig H  — Target distribution (train vs test)
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

# ─────────────────────────────────────────────────────────────
# Import config for all paths
# ─────────────────────────────────────────────────────────────
from config import (
    INPUT_CSV,
    WEATHER_CSV,
    FEATURES_CSV,
    PLOTS_DIR,
    TRAIN_RATIO,
    PROJECT_ROOT,
)

# Colors
BLUE = "#2196F3"
RED = "#EF5350"
GREEN = "#4CAF50"
ORANGE = "#FF9800"
PURPLE = "#9C27B0"
TEAL = "#009688"

# Muted steel-blue used for Figures C and D (ACF + Energy vs Temperature)
STEEL_BLUE = "#4682B4"

SEASON_COLORS = {
    "Winter (Dec–Feb)": "#5C7CFA",
    "Spring (Mar–May)": "#69DB7C", 
    "Summer (Jun–Aug)": "#FF6B6B",
    "Autumn (Sep–Nov)": "#FAB005",
}

# ─────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})


# ─────────────────────────────────────────────────────────────
# FIND ENERGY FILE
# ─────────────────────────────────────────────────────────────
def find_energy_file():
    """Find the energy CSV file (handles different naming)"""
    candidates = [
        INPUT_CSV,
        PROJECT_ROOT / "etl" / "data" / "processed" / "final.csv",
        PROJECT_ROOT / "etl" / "data" / "processed" / "hourly_dataset.csv",
        PROJECT_ROOT / "etl" / "data" / "processed" / "step1_hourly_aggregated.csv",
    ]
    for path in candidates:
        if path.exists():
            print(f"  Found energy file: {path.name}")
            return path
    return None


# ─────────────────────────────────────────────────────────────
# ENSURE DATETIME COLUMN
# ─────────────────────────────────────────────────────────────
def ensure_datetime(df, name="dataframe"):
    """Ensure datetime column is proper datetime type"""
    if "datetime" not in df.columns:
        print(f"  Warning: No datetime column in {name}")
        return df
    
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df["datetime"] = pd.to_datetime(df["datetime"])
        print(f"  Converted datetime in {name}")
    
    return df


# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
def load_data():
    print("[0] Loading data...")

    # Try to load features
    df = None
    if FEATURES_CSV.exists():
        df = pd.read_csv(FEATURES_CSV, parse_dates=["datetime"])
        df = ensure_datetime(df, "features.csv")
        print(f"    features.csv: {df.shape[0]:,} rows, {df.shape[1]} columns")
    else:
        print(f"    Warning: {FEATURES_CSV} not found")
    
    # Load raw energy
    energy_path = find_energy_file()
    raw = None
    if energy_path:
        raw = pd.read_csv(energy_path, parse_dates=["datetime"])
        raw = ensure_datetime(raw, "energy.csv")
        raw = raw.sort_values("datetime").reset_index(drop=True)
        print(f"    energy data: {raw.shape[0]:,} rows")
        
        # If no features, create basic ones from raw
        if df is None and raw is not None:
            df = create_basic_features(raw)
    
    # Load weather
    if WEATHER_CSV.exists():
        wdf = pd.read_csv(WEATHER_CSV, parse_dates=["datetime"])
        wdf = ensure_datetime(wdf, "weather.csv")
        if wdf["datetime"].dt.tz is not None:
            wdf["datetime"] = wdf["datetime"].dt.tz_localize(None)
        wdf = wdf.sort_values("datetime").reset_index(drop=True)
        print(f"    weather data: {wdf.shape[0]:,} rows")
    else:
        print(f"    Warning: Weather file not found at {WEATHER_CSV}")
        wdf = pd.DataFrame()

    return df, raw, wdf


def create_basic_features(raw):
    """Create minimal features from raw energy data"""
    print("    Creating basic features from energy data...")
    
    df = raw.copy()
    
    # Basic time features
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["dayofyear"] = df["datetime"].dt.dayofyear
    
    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # Target (shifted 24h ahead)
    df["target_energy_kwh"] = df["energy_kwh"].shift(-24)
    
    # Basic lags
    for lag in [1, 24, 48, 168]:
        df[f"lag_{lag}h"] = df["energy_kwh"].shift(lag)
    
    # Rolling statistics
    df["roll_mean_24h"] = df["energy_kwh"].rolling(24, min_periods=1).mean()
    
    # Boolean features
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_work_hours"] = ((df["dayofweek"] < 5) & (df["hour"] >= 8) & (df["hour"] <= 18)).astype(int)
    
    # Remove rows with NaN target
    df = df.dropna(subset=["target_energy_kwh"])
    
    print(f"    Created features: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────────────────────
# FIG A — Hourly energy profile
# ─────────────────────────────────────────────────────────────
def fig_a_hourly_profile(df, raw):
    print("[A] Hourly energy profile...")
    
    if raw is None or raw.empty:
        print("    Skipping: no raw energy data")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: overall mean by hour
    ax = axes[0]
    by_hour = raw.groupby(raw["datetime"].dt.hour)["energy_kwh"].mean()
    ax.bar(by_hour.index, by_hour.values, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Hour of Day", fontsize=11)
    ax.set_ylabel("Mean Energy Consumption (kWh)", fontsize=11)
    ax.set_title("Figure A1. Hourly Energy Profile (All Days)", fontsize=12, fontweight="bold")
    ax.set_xticks(range(0, 24, 2))
    if not by_hour.empty:
        peak_h = by_hour.idxmax()
        ax.annotate(f"Peak: {by_hour[peak_h]:.2f} kWh",
                    xy=(peak_h, by_hour[peak_h]),
                    xytext=(peak_h + 2, by_hour[peak_h] + 0.05),
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1),
                    fontsize=9, color=RED)

    # Right: work vs non-work day profiles
    ax = axes[1]
    df_tmp = raw.copy()
    df_tmp["hour"] = df_tmp["datetime"].dt.hour
    df_tmp["is_work"] = ((df_tmp["datetime"].dt.dayofweek < 5) & 
                          (df_tmp["hour"] >= 8) & (df_tmp["hour"] <= 18)).astype(int)
    
    for label, mask, color in [
        ("Work Hours (8-18)", df_tmp["is_work"] == 1, BLUE),
        ("Non-Work Hours", df_tmp["is_work"] == 0, ORANGE),
    ]:
        profile = df_tmp[mask].groupby("hour")["energy_kwh"].mean()
        ax.plot(profile.index, profile.values, label=label, color=color, lw=2, marker="o", markersize=4, alpha=0.8)
    ax.set_xlabel("Hour of Day", fontsize=11)
    ax.set_ylabel("Mean Energy (kWh)", fontsize=11)
    ax.set_title("Figure A2. Work Hours vs Non-Work Hours Profile", fontsize=12, fontweight="bold")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = PLOTS_DIR / "figA_hourly_profile.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    Saved: {path.name}")


# ─────────────────────────────────────────────────────────────
# FIG B — Correlation matrix
# ─────────────────────────────────────────────────────────────
def fig_b_correlation_matrix(df):
    print("[B] Correlation matrix...")
    
    if df is None or df.empty:
        print("    Skipping: no data")
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Find the target column (try both names)
    target_col = None
    if "target_energy_kwh" in numeric_cols:
        target_col = "target_energy_kwh"
    elif "target_24h" in numeric_cols:
        target_col = "target_24h"
    else:
        print("    Skipping: no target column found")
        return
    
    # Select most relevant columns for correlation
    priority_cols = [target_col, "lag_1h", "lag_24h", "lag_48h", "lag_168h", "lag_336h",
                     "roll_mean_24h", "roll_std_24h", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
                     "month_sin", "month_cos", "is_weekend", "is_work", "f_temp", "f_cdd_18"]
    
    corr_cols = [c for c in priority_cols if c in numeric_cols]
    # Add other columns that have high correlation with target
    if len(corr_cols) < 20 and len(numeric_cols) > 20:
        target_corr = df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
        for col in target_corr.index[1:21]:
            if col not in corr_cols and col != target_col:
                corr_cols.append(col)
    
    if len(corr_cols) < 2:
        print("    Skipping: insufficient columns")
        return

    corr_mat = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(corr_mat.values, cmap="RdBu_r", norm=norm, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Correlation", rotation=270, labelpad=15)

    ax.set_xticks(range(len(corr_cols)))
    ax.set_yticks(range(len(corr_cols)))
    ax.set_xticklabels(corr_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr_cols, fontsize=8)

    # Add correlation values for readability
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            val = corr_mat.values[i, j]
            if abs(val) > 0.3:
                text_color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=text_color)

    ax.set_title(f"Figure B. Feature Correlation Matrix\n(Highlighting {target_col} correlations)", 
                 fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    path = PLOTS_DIR / "figB_correlation_matrix.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    Saved: {path.name}")


# ─────────────────────────────────────────────────────────────
# FIG C — Autocorrelation function
# ─────────────────────────────────────────────────────────────
def fig_c_autocorrelation(raw):
    print("[C] Autocorrelation function...")
    
    if raw is None or raw.empty or "energy_kwh" not in raw.columns:
        print("    Skipping: no energy data")
        return

    series = raw["energy_kwh"].dropna().values
    max_lag = min(200, len(series) // 4)
    if max_lag < 24:
        print(f"    Skipping: insufficient data (len={len(series)})")
        return
        
    e_cent = series - series.mean()
    var_e = np.var(e_cent)
    if var_e < 1e-10:
        print("    Skipping: zero variance")
        return
        
    acf = np.array([1.0] + [
        float(np.mean(e_cent[k:] * e_cent[:-k]) / var_e)
        for k in range(1, max_lag + 1)
    ])
    conf = 1.96 / np.sqrt(len(series))

    fig, ax = plt.subplots(figsize=(16, 5))
    lags = np.arange(max_lag + 1)
    colors = STEEL_BLUE

    ax.bar(lags, acf, color=colors, alpha=0.75, width=0.8)
    ax.axhline(conf, color="gray", lw=1.5, ls="--", alpha=0.7, label=f"95% CI (±{conf:.3f})")
    ax.axhline(-conf, color="gray", lw=1.5, ls="--", alpha=0.7)
    ax.axhline(0, color="black", lw=0.8)

    # Annotate significant lags
    for lag in [24, 48, 72, 96, 120, 144, 168]:
        if lag < len(acf) and abs(acf[lag]) > conf:
            ax.annotate(f"lag {lag}h", xy=(lag, acf[lag]),
                        xytext=(lag + 3, acf[lag] + 0.05 if acf[lag] > 0 else acf[lag] - 0.1),
                        arrowprops=dict(arrowstyle="->", color=STEEL_BLUE, lw=1),
                        fontsize=9, color=STEEL_BLUE, fontweight="bold")

    ax.set_xlabel("Lag (hours)", fontsize=11)
    ax.set_ylabel("Autocorrelation", fontsize=11)
    ax.set_title("Figure C. Autocorrelation Function (ACF) — Hourly Energy Consumption", 
                 fontsize=12, fontweight="bold")
    ax.set_xlim(-2, max_lag + 2)
    ax.set_ylim(-0.6, 1.05)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = PLOTS_DIR / "figC_autocorrelation.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    Saved: {path.name}")


# ─────────────────────────────────────────────────────────────
# FIG D — Energy vs temperature
# ─────────────────────────────────────────────────────────────
def fig_d_energy_vs_temperature(df, wdf, raw):
    print("[D] Energy vs temperature...")
    
    if wdf is None or wdf.empty or "temperature" not in wdf.columns:
        print("    Skipping: no temperature data")
        return
    
    if raw is None or raw.empty:
        print("    Skipping: no energy data")
        return
        
    merged = raw.merge(wdf[["datetime", "temperature"]], on="datetime", how="inner")
    merged = merged.dropna(subset=["temperature", "energy_kwh"])

    if merged.empty:
        print("    Skipping: no merged data")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Scatter of raw points
    ax.scatter(merged["temperature"], merged["energy_kwh"],
               alpha=0.3, s=5, color=STEEL_BLUE, edgecolors="none", rasterized=True)

    ax.set_xlabel("Temperature at t (°C)", fontsize=11)
    ax.set_ylabel("Energy Consumption (kWh)", fontsize=11)
    ax.set_title("Figure D. Energy vs Temperature", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = PLOTS_DIR / "figD_energy_vs_temperature.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    Saved: {path.name}")


# ─────────────────────────────────────────────────────────────
# FIG E — Weather correlations
# ─────────────────────────────────────────────────────────────
def fig_e_weather_correlations(df, wdf):
    print("[E] Weather correlations with target...")
    
    if df is None or df.empty:
        print("    Skipping: no data")
        return
    
    # Find the target column
    target_col = None
    if "target_energy_kwh" in df.columns:
        target_col = "target_energy_kwh"
    elif "target_24h" in df.columns:
        target_col = "target_24h"
    
    if target_col is None:
        print("    Skipping: no target column found")
        return
    
    if wdf is None or wdf.empty:
        print("    Skipping: no weather data")
        return

    df_w = df.merge(wdf, on="datetime", how="left")
    target = pd.to_numeric(df_w[target_col], errors='coerce').values

    # Include weather forecast features if available
    weather_cols = []
    for col in ["temperature", "humidity", "apparent_temperature", "dew_point", "wind_speed", "cloud_cover"]:
        if col in df_w.columns:
            weather_cols.append(col)
    # Add forecast columns
    for col in ["f_temp", "f_humidity", "f_cloud_cover", "f_apparent_temp"]:
        if col in df_w.columns:
            weather_cols.append(col)
    
    results = []
    for col in weather_cols:
        vals = pd.to_numeric(df_w[col], errors='coerce').values
        valid_mask = ~np.isnan(vals) & ~np.isnan(target)
        if valid_mask.sum() < 50:
            continue
        vals_clean = vals[valid_mask]
        target_clean = target[valid_mask]
        try:
            pr, _ = pearsonr(vals_clean, target_clean)
            sr, _ = spearmanr(vals_clean, target_clean)
            results.append({"label": col, "pearson": pr, "spearman": sr})
        except:
            continue

    if not results:
        print("    Skipping: no valid correlations")
        return

    results = sorted(results, key=lambda x: abs(x["spearman"]), reverse=True)
    labels = [r["label"] for r in results]
    pearson = [r["pearson"] for r in results]
    spearman = [r["spearman"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pearson
    ax = axes[0]
    colors_pearson = [GREEN if p > 0 else RED for p in pearson]
    y = range(len(results))
    ax.barh(y, pearson, color=colors_pearson, alpha=0.8, height=0.6)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(0.2, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax.axvline(-0.2, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Pearson r", fontsize=11)
    ax.set_title("Figure E1. Pearson Correlation: Weather → Target", fontsize=11, fontweight="bold")
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(pearson):
        ax.text(v + (0.02 if v > 0 else -0.12), i, f"{v:.3f}", va='center', fontsize=8)

    # Spearman
    ax = axes[1]
    colors_spearman = [GREEN if s > 0 else RED for s in spearman]
    ax.barh(y, spearman, color=colors_spearman, alpha=0.8, height=0.6)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(0.2, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax.axvline(-0.2, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Spearman ρ", fontsize=11)
    ax.set_title("Figure E2. Spearman Correlation: Weather → Target", fontsize=11, fontweight="bold")
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(spearman):
        ax.text(v + (0.02 if v > 0 else -0.12), i, f"{v:.3f}", va='center', fontsize=8)

    plt.suptitle("Weather Feature Correlations with 24h-ahead Energy Target", 
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = PLOTS_DIR / "figE_weather_correlations.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    Saved: {path.name}")


# ─────────────────────────────────────────────────────────────
# FIG F — Seasonal profiles
# ─────────────────────────────────────────────────────────────
def fig_f_seasonal_profiles(df):
    print("[F] Seasonal energy profiles...")
    
    if df is None or df.empty:
        print("    Skipping: no data")
        return

    # Find the target column
    target_col = None
    if "target_energy_kwh" in df.columns:
        target_col = "target_energy_kwh"
    elif "target_24h" in df.columns:
        target_col = "target_24h"
    else:
        print(f"    Skipping: no target column found")
        return

    def get_season(m):
        if m in [12, 1, 2]: return "Winter (Dec–Feb)"
        if m in [3, 4, 5]: return "Spring (Mar–May)"
        if m in [6, 7, 8]: return "Summer (Jun–Aug)"
        return "Autumn (Sep–Nov)"

    df = df.copy()
    month_col = "month" if "month" in df.columns else ("t_month" if "t_month" in df.columns else None)
    if month_col is None:
        print("    Skipping: no month column")
        return
        
    df["season"] = df[month_col].apply(get_season)
    
    # Get hour column
    hour_col = "hour" if "hour" in df.columns else ("t_hour" if "t_hour" in df.columns else None)
    if hour_col is None:
        print("    Skipping: no hour column")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Hourly profiles by season
    ax = axes[0]
    for season, color in SEASON_COLORS.items():
        sub = df[df["season"] == season]
        if len(sub) < 50:
            continue
        profile = sub.groupby(hour_col)[target_col].mean()
        ax.plot(profile.index, profile.values, label=season, color=color, lw=2, marker='.', markersize=3)
    ax.set_xlabel("Hour of Day", fontsize=11)
    ax.set_ylabel(f"Mean {target_col.replace('_', ' ').title()} (kWh)", fontsize=11)
    ax.set_title("Figure F1. Seasonal Hourly Energy Profiles", fontsize=12, fontweight="bold")
    ax.set_xticks(range(0, 24, 3))
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: Monthly averages
    ax = axes[1]
    monthly = df.groupby(month_col)[target_col].mean()
    season_of_month = {1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
                       6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn",
                       11: "Autumn", 12: "Winter"}
    bar_colors = []
    for m in monthly.index:
        s = season_of_month.get(m, "Winter")
        bar_colors.append(SEASON_COLORS.get(f"{s} (Dec–Feb)", BLUE))
    
    ax.bar(monthly.index, monthly.values, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel(f"Mean {target_col.replace('_', ' ').title()} (kWh)", fontsize=11)
    ax.set_title("Figure F2. Mean Energy by Month", fontsize=12, fontweight="bold")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = PLOTS_DIR / "figF_seasonal_profiles.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    Saved: {path.name}")


# ─────────────────────────────────────────────────────────────
# FIG G — Work vs non-work
# ─────────────────────────────────────────────────────────────
def fig_g_work_vs_nonwork(df, raw):
    print("[G] Work vs non-work profiles...")
    
    if raw is None or raw.empty:
        print("    Skipping: no raw data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Work hours vs non-work hours
    ax = axes[0]
    raw_copy = raw.copy()
    raw_copy["hour"] = raw_copy["datetime"].dt.hour
    raw_copy["dayofweek"] = raw_copy["datetime"].dt.dayofweek
    raw_copy["is_work_hour"] = ((raw_copy["dayofweek"] < 5) & 
                                 (raw_copy["hour"] >= 8) & 
                                 (raw_copy["hour"] <= 18)).astype(int)
    
    work_profiles = {}
    for label, mask, color in [
        ("Work Hours (8-18, Mon-Thu)", raw_copy["is_work_hour"] == 1, BLUE),
        ("Non-Work Hours", raw_copy["is_work_hour"] == 0, ORANGE),
    ]:
        profile = raw_copy[mask].groupby("hour")["energy_kwh"].mean()
        work_profiles[label] = profile
        ax.plot(profile.index, profile.values, label=label, color=color, lw=2, marker="o", ms=4, alpha=0.8)
    ax.set_xlabel("Hour of Day", fontsize=11)
    ax.set_ylabel("Mean Energy (kWh)", fontsize=11)
    ax.set_title("Figure G1. Work Hours vs Non-Work Hours Profile", fontsize=12, fontweight="bold")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add shaded region for work hours
    ax.axvspan(8, 18, alpha=0.1, color=BLUE, label="Work hours zone")

    # Right: Weekday vs Weekend
    ax = axes[1]
    raw_copy["is_weekend"] = (raw_copy["dayofweek"] >= 5).astype(int)
    
    for label, mask, color in [
        ("Weekday (Mon-Thu)", (raw_copy["dayofweek"] < 4) & (raw_copy["dayofweek"] >= 0), BLUE),
        ("Friday", raw_copy["dayofweek"] == 4, PURPLE),
        ("Weekend", raw_copy["dayofweek"] >= 5, ORANGE),
    ]:
        profile = raw_copy[mask].groupby("hour")["energy_kwh"].mean()
        if len(profile) > 0:
            ax.plot(profile.index, profile.values, label=label, color=color, lw=2, marker='.', ms=4, alpha=0.8)
    
    ax.set_xlabel("Hour of Day", fontsize=11)
    ax.set_ylabel("Mean Energy (kWh)", fontsize=11)
    ax.set_title("Figure G2. Daily Profiles by Day Type", fontsize=12, fontweight="bold")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = PLOTS_DIR / "figG_work_nonwork_profiles.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    Saved: {path.name}")


# ─────────────────────────────────────────────────────────────
# FIG H — Train/test distribution
# ─────────────────────────────────────────────────────────────
def fig_h_train_test_distribution(df):
    print("[H] Train vs test distribution...")
    
    if df is None or df.empty:
        print("    Skipping: no data")
        return
    
    # Find the target column
    target_col = None
    if "target_energy_kwh" in df.columns:
        target_col = "target_energy_kwh"
    elif "target_24h" in df.columns:
        target_col = "target_24h"
    else:
        print(f"    Skipping: missing target column")
        return
    
    # Ensure datetime is proper type
    if "datetime" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
            df["datetime"] = pd.to_datetime(df["datetime"])
    else:
        print("    Skipping: no datetime column")
        return

    # Sort by date and split chronologically
    df_sorted = df.sort_values("datetime")
    sp = int(len(df_sorted) * TRAIN_RATIO)
    y_tr = df_sorted[target_col].values[:sp]
    y_te = df_sorted[target_col].values[sp:]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Histogram
    ax = axes[0]
    bins = np.linspace(0, max(df_sorted[target_col].max(), 10), 50)
    ax.hist(y_tr, bins=bins, alpha=0.6, color=BLUE, label=f"Train (n={len(y_tr):,})", density=True, edgecolor='white')
    ax.hist(y_te, bins=bins, alpha=0.6, color=RED, label=f"Test (n={len(y_te):,})", density=True, edgecolor='white')
    ax.axvline(y_tr.mean(), color=BLUE, lw=2, ls="--", label=f"Train mean = {y_tr.mean():.2f}")
    ax.axvline(y_te.mean(), color=RED, lw=2, ls="--", label=f"Test mean = {y_te.mean():.2f}")
    ax.set_xlabel("Target Energy (kWh)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Figure H1. Target Distribution: Train vs Test", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Time series of target
    ax = axes[1]
    # Sample for plotting (if too many points)
    n_samples = min(1000, len(df_sorted))
    sample_idx = np.linspace(0, len(df_sorted)-1, n_samples, dtype=int)
    plot_df = df_sorted.iloc[sample_idx]
    
    train_mask = plot_df.index < sp
    test_mask = plot_df.index >= sp
    
    ax.scatter(plot_df["datetime"][train_mask], plot_df[target_col][train_mask], 
               color=BLUE, s=5, alpha=0.4, label="Train")
    ax.scatter(plot_df["datetime"][test_mask], plot_df[target_col][test_mask], 
               color=RED, s=5, alpha=0.4, label="Test")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Energy (kWh)", fontsize=11)
    ax.set_title("Figure H2. Target Time Series (Sampled)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.suptitle("Target Distribution Analysis — Chronological Train/Test Split (80/20)", 
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = PLOTS_DIR / "figH_train_test_distribution.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    Saved: {path.name}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  VISUALIZATION — 24h Energy Forecasting Features")
    print("=" * 65)

    # Create plots directory if it doesn't exist
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df, raw, wdf = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n  Generating 8 figures → {PLOTS_DIR}/")
    print("  " + "-" * 40)

    fig_a_hourly_profile(df, raw)
    fig_b_correlation_matrix(df)
    fig_c_autocorrelation(raw)
    fig_d_energy_vs_temperature(df, wdf, raw)
    fig_e_weather_correlations(df, wdf)
    fig_f_seasonal_profiles(df)
    fig_g_work_vs_nonwork(df, raw)
    fig_h_train_test_distribution(df)

    print(f"\n{'='*65}")
    print(f"  ALL FIGURES SAVED → {PLOTS_DIR}/")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()