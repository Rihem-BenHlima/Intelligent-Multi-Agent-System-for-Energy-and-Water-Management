"""
generate_figures.py — Pipeline Visualization
=============================================
Generates diagnostic figures for each ETL stage:
  - Fig 2: Hourly coverage and meter regimes
  - Fig 3: Before/after comparison for gap imputation
  - Fig 4: Domain context validation (work hours, seasons, weekends)
  - Fig 5: Full clean timeline of hourly energy consumption
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import STEP1_OUTPUT, STEP2_OUTPUT, STEP3_OUTPUT, PLOTS_DIR

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def fig1_pipeline_overview():
    """Pipeline overview diagram placeholder."""
    pass


def fig2_coverage_and_regimes():
    """Step 1: Coverage timeline showing meter regimes."""
    df = pd.read_csv(STEP1_OUTPUT, parse_dates=["datetime"])

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    ax = axes[0]
    ax.plot(df["datetime"], df["coverage"], alpha=0.6, linewidth=0.5)
    ax.axhline(0.35, color="r", linestyle="--", label="threshold")
    ax.set_ylabel("Coverage")
    ax.set_title("Step 1: Hourly Coverage & Meter Regimes")
    ax.legend()

    ax = axes[1]
    ax.plot(df["datetime"], df["energy_kwh"], alpha=0.6, linewidth=0.5)
    ax.set_ylabel("kWh")
    ax.set_xlabel("Date")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig2_coverage_regimes.png", dpi=300)
    plt.close()
    print("Saved fig2_coverage_regimes.png")


def fig3_imputation_comparison():
    """Step 2: Before/after imputation on a long gap."""
    df1 = pd.read_csv(STEP1_OUTPUT, parse_dates=["datetime"])
    df2 = pd.read_csv(STEP2_OUTPUT, parse_dates=["datetime"])

    # Identify contiguous gaps and zoom into the longest contiguous gap
    nan_mask = df1["energy_kwh"].isna()
    gap = df1[nan_mask]
    if len(gap) > 0:
        gap_grp = (nan_mask != nan_mask.shift()).cumsum()

        # Pick the longest contiguous gap (most illustrative for the figure)
        gap_runs = df1[nan_mask].groupby(gap_grp[nan_mask])
        longest_id = gap_runs.size().idxmax()
        this_gap = gap_runs.get_group(longest_id)

        gap_start = this_gap["datetime"].min()
        gap_end   = this_gap["datetime"].max()
        sample_date = this_gap["datetime"].iloc[len(this_gap)//2]

        start = gap_start - pd.Timedelta(days=7)
        end   = gap_end + pd.Timedelta(days=7)

        s1 = df1[(df1["datetime"] >= start) & (df1["datetime"] <= end)]
        s2 = df2[(df2["datetime"] >= start) & (df2["datetime"] <= end)]

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(s1["datetime"], s1["energy_kwh"], "o-", alpha=0.5, label="Before (gaps)", markersize=2)
        ax.plot(s2["datetime"], s2["energy_kwh"], "-", alpha=0.8, label="After imputation", linewidth=1)
        ax.axvspan(gap_start, gap_end, alpha=0.2, color="blue", label="Imputed region")
        ax.set_ylabel("kWh")
        ax.set_title("Step 2: Context-Aware Imputation with Noise")
        ax.legend()

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "fig3_imputation.png", dpi=300)
        plt.close()
        print("Saved fig3_imputation.png")


def fig4_domain_context():
    """Step 2: Domain-aware context validation."""
    df = pd.read_csv(STEP2_OUTPUT, parse_dates=["datetime"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Hour profile by work hours
    ax = axes[0, 0]
    for is_work, grp in df.groupby("is_work_hours"):
        label = "Work hours (8-18)" if is_work else "Off hours"
        grp.groupby("hour")["energy_kwh"].mean().plot(ax=ax, label=label)
    ax.set_ylabel("Mean kWh")
    ax.set_title("Hourly Profile: Work vs Off Hours")
    ax.legend()

    # Season comparison
    ax = axes[0, 1]
    df.boxplot(column="energy_kwh", by="season", ax=ax)
    ax.set_title("Seasonal Distribution")
    plt.suptitle("")

    # Weekend vs weekday
    ax = axes[1, 0]
    df.boxplot(column="energy_kwh", by="is_weekend", ax=ax)
    ax.set_title("Weekday vs Weekend")
    ax.set_xticklabels(["Weekday", "Weekend"])
    plt.suptitle("")

    # Work hours distribution
    ax = axes[1, 1]
    df.boxplot(column="energy_kwh", by="is_work_hours", ax=ax)
    ax.set_title("Work Hours vs Off Hours")
    ax.set_xticklabels(["Off Hours", "Work Hours"])
    plt.suptitle("")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig4_domain_context.png", dpi=300)
    plt.close()
    print("Saved fig4_domain_context.png")


def fig5_full_timeline():
    """Final: Complete year timeline — clean dataset only, single color."""
    df3 = pd.read_csv(STEP3_OUTPUT, parse_dates=["datetime"])

    fig, ax = plt.subplots(figsize=(16, 5), dpi=300)

    # Single blue trace — final clean data only
    ax.plot(df3["datetime"], df3["energy_kwh"], 
            color="#1f77b4", alpha=0.85, linewidth=0.5)

    # Labels
    ax.set_ylabel("Energy (kWh)", fontsize=11, fontweight="medium")
    ax.set_xlabel("Date", fontsize=11, fontweight="medium")
    ax.set_title("Final Clean Hourly Energy Consumption — Full Year Timeline",
                 fontsize=12, fontweight="bold", pad=15)

    # Subtle grid
    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    # Clean spines
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#666666")

    ax.tick_params(axis="both", labelsize=9, colors="#333333")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig5_full_timeline.png", dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print("Saved fig5_full_timeline.png")

def run():
    print("=" * 60)
    print("  Generating Figures")
    print("=" * 60)

    fig2_coverage_and_regimes()
    fig3_imputation_comparison()
    fig4_domain_context()
    fig5_full_timeline()

    print(f"\n  All figures saved to {PLOTS_DIR}")


if __name__ == "__main__":
    run()