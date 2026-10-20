"""
Feature Engineering — Office Building Water Consumption
Generates temporal, calendar, lag, difference, rolling, and EWMA features
from cleaned water consumption data (no weather/seasonality features).
Outputs: features_engineering/output/features.csv
Run: python generate_features.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import ETL step3 output path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from etl.config import STEP3_OUTPUT

STEEL_BLUE = "#4682B4"


def plot_autocorrelation(series, out_path, max_lag=200,
                          annotate_lags=(24, 48, 72, 96, 120, 144, 168)):
    """Plot autocorrelation function (ACF) of hourly water consumption."""
    series = np.asarray(series, dtype=float)
    max_lag = min(max_lag, len(series) // 4)
    if max_lag < 24:
        print(f"  Skipping ACF plot: insufficient data (len={len(series)})")
        return

    c = series - series.mean()
    var = np.var(c)
    if var < 1e-10:
        print("  Skipping ACF plot: zero variance")
        return

    acf = np.array([1.0] + [
        float(np.mean(c[k:] * c[:-k]) / var) for k in range(1, max_lag + 1)
    ])
    conf = 1.96 / np.sqrt(len(series))
    lags = np.arange(max_lag + 1)

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "DejaVu Sans",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(lags, acf, color=STEEL_BLUE, alpha=0.75, width=0.8)
    ax.axhline(conf, color="gray", lw=1.5, ls="--", alpha=0.7, label=f"95% CI (±{conf:.3f})")
    ax.axhline(-conf, color="gray", lw=1.5, ls="--", alpha=0.7)
    ax.axhline(0, color="black", lw=0.8)

    for lag in annotate_lags:
        if lag < len(acf) and abs(acf[lag]) > conf:
            ax.annotate(f"lag {lag}h", xy=(lag, acf[lag]),
                        xytext=(lag + 3, acf[lag] + 0.05 if acf[lag] > 0 else acf[lag] - 0.1),
                        arrowprops=dict(arrowstyle="->", color=STEEL_BLUE, lw=1),
                        fontsize=9, color=STEEL_BLUE, fontweight="bold")

    ax.set_xlabel("Lag (hours)", fontsize=11)
    ax.set_ylabel("Autocorrelation", fontsize=11)
    ax.set_title("Autocorrelation Function (ACF) — Hourly Office Water Consumption",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(-2, max_lag + 2)
    ax.set_ylim(-0.6, 1.05)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved -> {out_path}")


def run():
    print("=" * 55)
    print(" FEATURE ENGINEERING (Lags, Rolling, Cyclical)")
    print("=" * 55)

    base_dir   = Path(__file__).resolve().parent
    input_file = STEP3_OUTPUT
    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "features.csv"

    plot_dir = base_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        print("Please run the ETL pipeline first (step1 -> step2 -> step3).")
        sys.exit(1)

    print(f"Loading: {input_file.name}...")
    df = pd.read_csv(input_file, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # 1. Autocorrelation function
    print("Generating autocorrelation plot...")
    plot_autocorrelation(df["consumption_l"], plot_dir / "autocorrelation.png")

    # 2. Cyclical temporal encoding
    print("Generating cyclical temporal features...")
    hour        = df["datetime"].dt.hour
    day_of_week = df["datetime"].dt.dayofweek
    week_of_year = df["datetime"].dt.isocalendar().week.astype(int)
    month        = df["datetime"].dt.month

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["day_sin"]  = np.sin(2 * np.pi * day_of_week / 7)
    df["day_cos"]  = np.cos(2 * np.pi * day_of_week / 7)
    df["week_sin"] = np.sin(2 * np.pi * week_of_year / 52)
    df["week_cos"] = np.cos(2 * np.pi * week_of_year / 52)
    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    df["hour"]         = hour
    df["day_of_week"]  = day_of_week
    df["week_of_year"] = week_of_year
    df["month"]        = month

    # 3. Binary calendar features
    print("Generating working hours and weekend flags...")
    df["is_working_hour"] = ((hour >= 8) & (hour <= 18) & (day_of_week < 5)).astype(int)
    df["is_weekend"]      = (day_of_week >= 5).astype(int)

    # 4. Lag features (up to 25h to compute 24h diff)
    print("Generating lag features...")
    for i in [1, 2, 3, 24, 25, 168]:
        df[f"lag_{i}h"] = df["consumption_l"].shift(i)

    # 5. Difference features (diff_1h: t-1 vs t-2, diff_24h: t-1 vs t-25)
    print("Generating difference features...")
    df["diff_1h"]  = df["lag_1h"] - df["lag_2h"]
    df["diff_24h"] = df["lag_1h"] - df["lag_25h"]
    df = df.drop(columns=["lag_25h"])

    # 6. Rolling statistics (shifted by 1 to prevent target leakage)
    print("Generating rolling window features...")
    base = df["lag_1h"]

    df["rolling_mean_3h"]   = base.rolling(window=3).mean()
    df["rolling_mean_6h"]   = base.rolling(window=6).mean()
    df["rolling_mean_24h"]  = base.rolling(window=24).mean()
    df["rolling_std_24h"]   = base.rolling(window=24).std()
    df["rolling_max_24h"]   = base.rolling(window=24).max()
    df["rolling_min_24h"]   = base.rolling(window=24).min()
    df["rolling_mean_168h"] = base.rolling(window=168).mean()
    df["rolling_std_168h"]  = base.rolling(window=168).std()

    # 7. Exponentially Weighted Moving Average
    print("Generating EWMA features...")
    df["ewma_6h"]  = base.ewm(span=6,  adjust=False).mean()
    df["ewma_24h"] = base.ewm(span=24, adjust=False).mean()

    # 8. Cleanup
    initial_rows = len(df)
    df = df.dropna().reset_index(drop=True)
    final_rows   = len(df)

    # 9. Working-schedule empirical probability (fit on train split only to avoid leakage)
    TRAIN_FRACTION = 0.70
    train_end = int(final_rows * TRAIN_FRACTION)
    train_slice = df.iloc[:train_end]
    door_prob = (
        (train_slice["consumption_l"] > 0).astype(float)
        .groupby([train_slice["day_of_week"], train_slice["hour"]])
        .mean()
        .rename("door_open_prob")
    )
    df = df.merge(door_prob, left_on=["day_of_week", "hour"], right_index=True, how="left")
    df["door_open_prob"] = df["door_open_prob"].fillna(door_prob.mean())

    # Exclude non-feature columns.
    feature_cols = [c for c in df.columns
                    if c not in ["datetime", "consumption_l"]]

    print(f"\nFeature generation complete!")
    print(f"Rows dropped due to lags/rolling : {initial_rows - final_rows:,}")
    print(f"Final dataset shape              : {df.shape}")
    print(f"\nAll generated features ({len(feature_cols)}):")
    for col in feature_cols:
        print(f"  - {col}")

    df.to_csv(output_file, index=False)
    print(f"\nSaved → {output_file}")


if __name__ == "__main__":
    run()