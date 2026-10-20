"""
STEP 2 — Gap classification & tiered imputation.
  Tier 1  <=3h                        linear interpolation
  Tier 2  4-24h                       (dow, hour) profile + noise
  Tier 3  25h - MAX_IMPUTE_GAP_HOURS  (dow, hour) profile + noise
  Tier 4  > MAX_IMPUTE_GAP_HOURS      real-week block bootstrap
"""
import pandas as pd
import numpy as np
from config import HOUSEHOLD_ID, TIER1_MAX, TIER2_MAX, MAX_IMPUTE_GAP_HOURS, NOISE_FACTOR


def run(step1_df):
    df = step1_df.sort_values("datetime").reset_index(drop=True)
    df["consumption_l_pre"] = df["consumption_l"].copy()
    df["hour"] = df["datetime"].dt.hour
    df["dow"] = df["datetime"].dt.dayofweek

    df["_is_nan"] = df["consumption_l"].isna()
    df["_gap_id"] = (df["_is_nan"] != df["_is_nan"].shift()).cumsum()
    df["_gap_len"] = df.groupby("_gap_id")["_is_nan"].transform("sum")
    df["_gap_len"] = df["_gap_len"].where(df["_is_nan"], 0)

    df["_tier"] = 0
    df.loc[df["_is_nan"] & (df["_gap_len"] <= TIER1_MAX), "_tier"] = 1
    df.loc[df["_is_nan"] & (df["_gap_len"] > TIER1_MAX) & (df["_gap_len"] <= TIER2_MAX), "_tier"] = 2
    df.loc[df["_is_nan"] & (df["_gap_len"] > TIER2_MAX) & (df["_gap_len"] <= MAX_IMPUTE_GAP_HOURS), "_tier"] = 3
    df.loc[df["_is_nan"] & (df["_gap_len"] > MAX_IMPUTE_GAP_HOURS), "_tier"] = 4

    t1 = (df["_tier"] == 1).sum()
    t2 = (df["_tier"] == 2).sum()
    t3 = (df["_tier"] == 3).sum()
    t4 = (df["_tier"] == 4).sum()
    hh = HOUSEHOLD_ID
    print(f"  [{hh}] Tier1(<=3h): {t1:,} | Tier2(4-24h): {t2:,} | "
          f"Tier3(25h-{MAX_IMPUTE_GAP_HOURS}h): {t3:,} | "
          f"Tier4(>{MAX_IMPUTE_GAP_HOURS}h, real-week bootstrap): {t4:,}")

    full_interp = df["consumption_l"].interpolate(method="linear", limit_area="inside")
    tier1_mask = df["_tier"] == 1
    df.loc[tier1_mask, "consumption_l"] = full_interp[tier1_mask]
    df["consumption_l"] = df["consumption_l"].clip(lower=0)

    # NOTE: profiles (and the Tier-4 week pool below) are built from the
    # FULL series, so a gap inside the training period is filled using
    # stats that include later hours. Negligible in practice (Larry: 56h,
    # 0.3%) and gap_tier/is_imputed/fill_method are kept in the output so
    # downstream evaluation can exclude imputed hours if needed.
    known = df[df["consumption_l"].notna()]
    prof = (known.groupby(["dow", "hour"])["consumption_l"]
            .agg(["mean", "std"]).reset_index())
    prof.columns = ["dow", "hour", "p_mean", "p_std"]
    prof["p_std"] = prof["p_std"].fillna(prof["p_mean"] * 0.15).clip(lower=0.01)

    prof_hr = (known.groupby("hour")["consumption_l"]
               .agg(["mean", "std"]).reset_index())
    prof_hr.columns = ["hour", "h_mean", "h_std"]
    prof_hr["h_std"] = prof_hr["h_std"].fillna(prof_hr["h_mean"] * 0.15).clip(lower=0.01)

    df = df.merge(prof, on=["dow", "hour"], how="left")
    df = df.merge(prof_hr, on="hour", how="left")
    df["p_mean"] = df["p_mean"].fillna(df["h_mean"])
    df["p_std"] = df["p_std"].fillna(df["h_std"]).fillna(0.1)

    rng = np.random.default_rng(2024)

    for tier in (2, 3):
        mask = (df["_tier"] == tier) & df["consumption_l"].isna()
        n = mask.sum()
        if n == 0:
            continue
        mu = df.loc[mask, "p_mean"].values
        sd = df.loc[mask, "p_std"].values
        df.loc[mask, "consumption_l"] = np.clip(mu + rng.normal(0, 1, n) * sd * NOISE_FACTOR, 0, None)

    df["fill_method"] = np.where(df["_tier"] == 0, "measured",
                          np.where(df["_tier"] == 1, "linear_interp",
                          np.where(df["_tier"].isin([2, 3]), "profile_noise", "none")))

    t4_mask = df["_tier"] == 4
    if t4_mask.any():
        iso = df["datetime"].dt.isocalendar()
        df["_week_key"] = list(zip(iso["year"], iso["week"]))
        df["_pos_in_week"] = df["dow"] * 24 + df["hour"]

        week_completeness = df.groupby("_week_key")["consumption_l_pre"].agg(
            n="size", n_real=lambda s: s.notna().sum())
        full_real_weeks = week_completeness[
            (week_completeness["n"] == 168) & (week_completeness["n_real"] == 168)
        ].index.tolist()

        source_lookup = {}
        for wk in full_real_weeks:
            wk_df = df[df["_week_key"] == wk].set_index("_pos_in_week")["consumption_l_pre"]
            source_lookup[wk] = wk_df

        rng = np.random.default_rng(2024)
        target_weeks = df.loc[t4_mask, "_week_key"].unique().tolist()
        if full_real_weeks:
            chosen = {wk: full_real_weeks[i] for wk, i in
                      zip(target_weeks, rng.integers(0, len(full_real_weeks), size=len(target_weeks)))}
        else:
            chosen = {}

        for wk, rows in df.loc[t4_mask].groupby("_week_key"):
            if wk not in chosen:
                continue
            src = source_lookup[chosen[wk]]
            df.loc[rows.index, "consumption_l"] = rows["_pos_in_week"].map(src).values
            df.loc[rows.index, "fill_method"] = "real_week_bootstrap"

        df = df.drop(columns=["_week_key", "_pos_in_week"])

    still_nan = df["consumption_l"].isna()
    if still_nan.any():
        df["consumption_l"] = df["consumption_l"].ffill().bfill()
        df.loc[still_nan, "fill_method"] = df.loc[still_nan, "fill_method"].where(
            df.loc[still_nan, "fill_method"] != "none", "ffill_bfill_fallback")

    df["consumption_l"] = df["consumption_l"].clip(lower=0)

    remaining = df["consumption_l"].isna().sum()
    print(f"  [{hh}] Tier 4 filled via real-week block bootstrap: {t4_mask.sum():,} "
          f"| NaN remaining after imputation: {remaining:,}")

    out = df[["datetime", "consumption_l", "consumption_l_pre",
              "hour", "dow", "_tier", "fill_method"]].copy()
    out = out.rename(columns={"_tier": "gap_tier"})
    return out


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from config import STEP2_OUTPUT
    import step1_clean_aggregate as step1

    print("=" * 60)
    print("  STEP 2 — Gap Classification & Tiered Imputation")
    print("=" * 60)

    s1 = step1.run()
    out = run(s1)

    out.to_csv(STEP2_OUTPUT, index=False)
    print(f"\n  Saved -> {STEP2_OUTPUT}")
    print(f"  Shape: {out.shape}")