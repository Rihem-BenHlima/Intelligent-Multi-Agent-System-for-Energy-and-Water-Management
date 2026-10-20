"""STEP 1 — spike removal and zero-run gap flagging."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (RAW_INPUT, STEP1_OUTPUT, PLOT_DIR, HOUSEHOLD_ID,
                     RATE_CAP_L_PER_H, ZERO_RUN_GAP_HOURS, SAMPLE_WINDOW_DAYS,
                     SERIES_START, MAX_SYNTHETIC_FRACTION,
                     SPIKE_Z_THRESHOLD, SPIKE_MIN_ABS_L_PER_H)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 10,
})

SEP = "=" * 60


def run():
    print(SEP)
    print("  STEP 1 — Clean & Spike Removal (single household)")
    print(SEP)

    print(f"\n  Loading: {RAW_INPUT}")
    df = pd.read_csv(RAW_INPUT, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"  Rows        : {len(df):,}")
    print(f"  Range       : {df['datetime'].min()} -> {df['datetime'].max()}")

    if SERIES_START is not None:
        n_before = len(df)
        df = df[df["datetime"] >= pd.Timestamp(SERIES_START)].reset_index(drop=True)
        print(f"  SERIES_START={SERIES_START} -> dropped {n_before - len(df):,} "
              f"leading rows, {len(df):,} remain")

    raw_nan_frac = df["consumption_l"].isna().mean()
    print(f"  Raw NaN fraction: {raw_nan_frac*100:.1f}%")
    if raw_nan_frac > MAX_SYNTHETIC_FRACTION:
        raise ValueError(
            f"ABORT: {raw_nan_frac*100:.1f}% of '{HOUSEHOLD_ID}' is missing in the "
            f"raw file (limit {MAX_SYNTHETIC_FRACTION*100:.0f}%). Imputing this much "
            f"data would fabricate the training set. Truncate the series with "
            f"config.SERIES_START or use a different building.")

    full_idx = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="h")
    missing_slots = len(full_idx) - len(df)
    dup = df["datetime"].duplicated().sum()
    print(f"\n  Timeline check -> expected {len(full_idx):,} hourly slots, "
          f"got {len(df):,}, missing {missing_slots:,}, duplicates {dup:,}")
    if missing_slots != 0 or dup != 0:
        df = (df.set_index("datetime").reindex(full_idx)
                .rename_axis("datetime").reset_index())
        print("  -> reindexed onto a complete uniform timeline")

    df["consumption_l_prespike"] = df["consumption_l"].copy()

    # Hard-cap: physically impossible readings (sensor error)
    hardcap_mask = df["consumption_l"] > RATE_CAP_L_PER_H

    # Statistical: real-but-anomalous readings RATE_CAP is far too high to ever
    # catch (e.g. a genuine multi-day demand spike). Flagged only if BOTH
    # extreme for the (day-of-week, hour) slot AND large in absolute terms —
    # the absolute floor avoids false positives on near-zero hours, where a
    # single small pulse would otherwise produce a huge z-score.
    hour = df["datetime"].dt.hour
    dow = df["datetime"].dt.dayofweek
    prof_mean = df.groupby([dow, hour])["consumption_l"].transform("mean")
    prof_std = df.groupby([dow, hour])["consumption_l"].transform("std")
    z = (df["consumption_l"] - prof_mean) / prof_std.replace(0, np.nan)
    zscore_mask = ((z > SPIKE_Z_THRESHOLD) & (df["consumption_l"] > SPIKE_MIN_ABS_L_PER_H)).fillna(False)

    spike_mask = hardcap_mask | zscore_mask
    print(f"\n  Spike readings: {hardcap_mask.sum():,} > {RATE_CAP_L_PER_H} L/h (hard cap) "
          f"+ {zscore_mask.sum():,} statistical (z>{SPIKE_Z_THRESHOLD} & "
          f">{SPIKE_MIN_ABS_L_PER_H} L/h) = {spike_mask.sum():,} total -> NaN")
    df.loc[spike_mask, "consumption_l"] = np.nan

    print("\n  --- Zero-consumption accounting ---")
    total_zero = (df["consumption_l"] == 0).sum()
    print(f"  Total exact-zero hours        : {total_zero:,}  "
          f"({total_zero/len(df)*100:.1f}% of all hours)")

    z = (df["consumption_l"] == 0).astype(int)
    run_id = (z != z.shift()).cumsum()
    df["_zero_run_len"] = df.groupby(run_id)["consumption_l"].transform(
        lambda s: len(s) if (s == 0).all() else 0
    )

    long_zero_mask = (df["consumption_l"] == 0) & (df["_zero_run_len"] >= ZERO_RUN_GAP_HOURS)
    n_long_zero_hours = long_zero_mask.sum()
    n_long_zero_runs = df.loc[long_zero_mask, "_zero_run_len"].groupby(
        run_id[long_zero_mask]).ngroups if long_zero_mask.any() else 0
    print(f"  Zero-runs >= {ZERO_RUN_GAP_HOURS}h (flagged as gaps) : "
          f"{n_long_zero_hours:,} hours across {n_long_zero_runs} run(s) -> NaN")
    print(f"  Zero-runs < {ZERO_RUN_GAP_HOURS}h (kept as genuine non-use): "
          f"{total_zero - n_long_zero_hours:,} hours")

    df["consumption_l_prezero"] = df["consumption_l"].copy()
    df.loc[long_zero_mask, "consumption_l"] = np.nan
    df = df.drop(columns=["_zero_run_len"])

    total_nan = df["consumption_l"].isna().sum()
    print(f"\n  Total NaN after Step 1 (spikes + flagged zero-runs): {total_nan:,} "
          f"({total_nan/len(df)*100:.1f}%)")

    if total_nan / len(df) > MAX_SYNTHETIC_FRACTION:
        raise ValueError(
            f"ABORT: {total_nan/len(df)*100:.1f}% of hours are NaN after Step 1 "
            f"cleaning (limit {MAX_SYNTHETIC_FRACTION*100:.0f}%). Review "
            f"RATE_CAP_L_PER_H / ZERO_RUN_GAP_HOURS or truncate via SERIES_START "
            f"— do not let Step 2 impute this much of the series.")

    out = df[["datetime", "consumption_l", "consumption_l_prespike"]].copy()
    out.to_csv(STEP1_OUTPUT, index=False)
    print(f"\n  Saved -> {STEP1_OUTPUT}")
    print(f"  Shape: {out.shape}")

    make_plot(df, spike_mask, long_zero_mask)

    return out


def make_plot(df, spike_mask, long_zero_mask):
    Path(PLOT_DIR).mkdir(exist_ok=True)

    start = df["datetime"].min()
    end = start + pd.Timedelta(days=SAMPLE_WINDOW_DAYS)
    win = df[(df["datetime"] >= start) & (df["datetime"] < end)]

    fig, axes = plt.subplots(3, 1, figsize=(13, 10))
    fig.suptitle(f"STEP 1 — Clean & Spike Removal — {HOUSEHOLD_ID}", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(df["datetime"], df["consumption_l_prespike"], color="#1f77b4", linewidth=0.5, zorder=1)
    if spike_mask.any():
        ax.scatter(df.loc[spike_mask, "datetime"], df.loc[spike_mask, "consumption_l_prespike"],
                   color="#ef4444", s=18, zorder=3, label=f"Spikes removed ({spike_mask.sum()})")
    if long_zero_mask.any():
        for t in df.loc[long_zero_mask, "datetime"]:
            ax.axvspan(t, t + pd.Timedelta(hours=1), color="#fca5a5", alpha=0.35, zorder=0)
    ax.axhline(0, color="#94a3b8", linewidth=0.6)
    ax.set_title("Full year — raw consumption, spikes marked, flagged zero-runs shaded", fontsize=11)
    ax.set_ylabel("Litres / hour")
    if spike_mask.any():
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(win["datetime"], win["consumption_l_prespike"], color="#cbd5e1", linewidth=1.2,
             label="Original", zorder=1)
    ax.plot(win["datetime"], win["consumption_l"], color="#0f766e", linewidth=1.4,
             label="After spike + zero-run gap removal", zorder=2)
    gap_mask_win = win["consumption_l"].isna()
    if gap_mask_win.any():
        for t in win.loc[gap_mask_win, "datetime"]:
            ax.axvspan(t, t + pd.Timedelta(hours=1), color="#fca5a5", alpha=0.35, zorder=0)
    ax.set_title(f"First {SAMPLE_WINDOW_DAYS} days — zoom view", fontsize=11)
    ax.set_ylabel("Litres / hour")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25)

    ax = axes[2]
    z = (df["consumption_l_prespike"] == 0).astype(int)
    run_id = (z != z.shift()).cumsum()
    runs = df.assign(_z=z, _rid=run_id).groupby("_rid").agg(
        is_zero=("_z", "all"), length=("_z", "size"))
    zero_lengths = runs.loc[runs.is_zero, "length"].values

    if len(zero_lengths) > 0:
        max_len = zero_lengths.max()
        if max_len > 0:
            bins = np.arange(0.5, max_len + 1.5, 1)
            ax.hist(zero_lengths, bins=bins, color="#f59e0b", edgecolor="white")
            ax.axvline(ZERO_RUN_GAP_HOURS - 0.5, color="#dc2626", linestyle="--", linewidth=1.2,
                       label=f"Gap threshold (≥{ZERO_RUN_GAP_HOURS}h flagged)")
            ax.set_title(f"Distribution of consecutive zero-consumption run lengths (n={len(zero_lengths)})", fontsize=11)
        else:
            ax.text(0.5, 0.5, "No zero runs found", transform=ax.transAxes, 
                   ha="center", va="center", fontsize=12)
    else:
        ax.text(0.5, 0.5, "No zero runs found", transform=ax.transAxes, 
               ha="center", va="center", fontsize=12)
    
    ax.set_xlabel("Run length (hours)")
    ax.set_ylabel("Count of runs")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = f"{PLOT_DIR}/step1_clean_aggregate.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Plot saved -> {out_path}")


if __name__ == "__main__":
    run()