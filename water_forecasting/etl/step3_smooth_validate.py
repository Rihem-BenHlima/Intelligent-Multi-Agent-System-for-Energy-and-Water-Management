"""
STEP 3 — Final validation. Modeling output is unsmoothed (exactly the
Step-2 imputed series) - a centered Gaussian smoothing is non-causal and
was previously leaking future values into lag features. A smoothed copy
is computed only for the presentation plot, never written to the CSV.
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import HOUSEHOLD_ID


def _smooth_with_nans(vals, sigma):
    """NaN-aware centered Gaussian smoothing - plot use only (non-causal)."""
    vals = vals.astype(float)
    mask = ~np.isnan(vals)
    filled = np.where(mask, vals, 0.0)
    sm_vals = gaussian_filter1d(filled, sigma=sigma)
    sm_mask = gaussian_filter1d(mask.astype(float), sigma=sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = sm_vals / sm_mask
    result[sm_mask < 1e-6] = np.nan
    result[~mask] = np.nan
    return result


def run(step2_df, sigma):
    """`sigma` is unused - kept for signature compatibility (smoothing is plot-only)."""
    df = step2_df.sort_values("datetime").reset_index(drop=True)
    df["is_imputed"] = df["gap_tier"].isin([1, 2, 3, 4])

    hh = HOUSEHOLD_ID
    nan_c = df["consumption_l"].isna().sum()
    neg_c = (df["consumption_l"] < 0).sum()
    tier4_c = (df["gap_tier"] == 4).sum()
    full_idx = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="h")
    missing_slots = len(full_idx) - len(df)

    print(f"  [{hh}] Final NaN: {nan_c:,} {'OK' if nan_c == 0 else 'FAIL'}")
    print(f"  [{hh}] Negative: {neg_c} {'OK' if neg_c == 0 else 'FAIL'} | "
          f"Missing slots: {missing_slots} {'OK' if missing_slots == 0 else 'FAIL'} | "
          f"Imputed hours: {df['is_imputed'].sum():,} ({df['is_imputed'].mean()*100:.1f}%) | "
          f"Tier-4 (real-analog back/forward fill) hours: {tier4_c:,} ({tier4_c/len(df)*100:.1f}%)")

    out = df[["datetime", "consumption_l", "gap_tier",
              "is_imputed", "fill_method"]].copy()
    return out, df


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from config import PLOT_DIR, SMOOTH_SIGMA, STEP3_OUTPUT
    import step1_clean_aggregate as step1
    import step2_detect_impute as step2

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "font.size": 10,
    })

    print("=" * 60)
    print("  STEP 3 — Smoothing & Final Validation")
    print("=" * 60)

    s1 = step1.run()
    s2 = step2.run(s1)
    out, df = run(s2, SMOOTH_SIGMA)

    out.to_csv(STEP3_OUTPUT, index=False)
    print(f"\n  Saved -> {STEP3_OUTPUT}")
    print(f"  Shape: {out.shape}")

    tier4_mask = df["gap_tier"] == 4

    fig, ax = plt.subplots(figsize=(13, 4.5))
    disp = np.clip(_smooth_with_nans(df["consumption_l"].values.astype(float),
                                     SMOOTH_SIGMA), 0, None)
    ax.plot(df["datetime"], disp, color="#1f77b4", linewidth=0.5)
    if tier4_mask.any():
        # shade each contiguous Tier-4 run separately, not one span over all of them
        run_id = (tier4_mask != tier4_mask.shift()).cumsum()
        n_h = int(tier4_mask.sum())
        first = True
        for _, seg in df[tier4_mask].groupby(run_id[tier4_mask]):
            ax.axvspan(seg["datetime"].min(),
                       seg["datetime"].max() + pd.Timedelta(hours=1),
                       color="#bbf7d0", alpha=0.4, zorder=0,
                       label=(f"Tier 4: real-week block bootstrap ({n_h:,}h)"
                              if first else None))
            first = False
        ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"STEP 3 — Final Validation — {HOUSEHOLD_ID} "
                 f"(display smoothed; CSV unsmoothed)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Litres/hour")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.25)

    plt.tight_layout()
    out_path = PLOT_DIR / "step3_smooth_validate.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Plot saved -> {out_path}")