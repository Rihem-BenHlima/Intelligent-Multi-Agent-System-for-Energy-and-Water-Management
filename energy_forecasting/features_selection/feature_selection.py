"""
feature_selection.py
====================
Feature selection for 24h-ahead building energy forecasting.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats as scipy_stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
import json

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
from config import (
    FEATURES_CSV, OUT_DIR,
    REPORT_TXT, FINAL_LIST_TXT, SELECTION_JSON, FINAL_FEAT_CSV,
)

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.15   # Validation ratio within training data for scoring candidates
RANDOM_SEED = 42
N_PLOT_TOP  = 30
EMBARGO_H   = 24     # Embargo horizon (hours) to eliminate label overlap at boundaries

# Feature groups
GROUP_COLORS = {
    "lag": "#1565C0",
    "rolling": "#0288D1",
    "calendar": "#2E7D32",
    "flag": "#66BB6A",
    "weather_obs": "#F57F17",
    "weather_fc": "#E65100",
    "interaction": "#7B1FA2",
    "profile": "#C62828",
    "other": "#90A4AE",
}


def get_group(col: str) -> str:
    if col.startswith("lag_"): return "lag"
    if any(x in col for x in ["roll_", "ewm_"]): return "rolling"
    if any(x in col for x in ["sin", "cos", "t_hour", "t_dow", "t_month", "t_doy", "hour", "month"]): return "calendar"
    if any(x in col for x in ["is_", "bridge_"]): return "flag"
    if "24h" in col and "lag" not in col and "roll" not in col: return "weather_fc"
    if any(x in col for x in ["_obs", "cdd_", "hdd_", "consec", "solar", "humid", "dew_", "temp_"]): return "weather_obs"
    if "_x_" in col: return "interaction"
    if "profile" in col: return "profile"
    return "other"


# ─────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────
def load_features():
    print(f"[1] Loading: {FEATURES_CSV}")
    
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"Features file not found: {FEATURES_CSV}")
    
    df = pd.read_csv(FEATURES_CSV, parse_dates=["datetime"])
    
    if df.empty:
        raise ValueError(f"Features file is empty: {FEATURES_CSV}")
    
    df = df.sort_values("datetime").reset_index(drop=True)
    
    print(f"   Loaded {len(df):,} rows")
    
    # Check datetime column
    if "datetime" not in df.columns:
        raise ValueError("No datetime column in features file")
    
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df["datetime"] = pd.to_datetime(df["datetime"])
        print("   Converted datetime column")
    
    # Identify feature columns
    META = {"datetime", "sn", "target_24h"}
    fcols = [c for c in df.columns 
             if c not in META 
             and pd.api.types.is_numeric_dtype(df[c])]
    
    print(f"   Found {len(fcols)} numeric feature columns")
    
    # Drop constant columns
    before = len(fcols)
    constant_cols = []
    for c in fcols:
        if df[c].std() < 1e-8:
            constant_cols.append(c)
    fcols = [c for c in fcols if c not in constant_cols]
    
    if constant_cols:
        print(f"   Dropped {len(constant_cols)} constant columns")
    
    if len(fcols) == 0:
        raise ValueError("No valid features found after removing constant columns")
    
    # Temporal split
    s = int(len(df) * TRAIN_RATIO)
    if s < 2 or s >= len(df):
        raise ValueError(f"Split index invalid: s={s}, len={len(df)}")
    
    X_tr = df[fcols].values[:s]
    y_tr = df["target_24h"].values[:s]
    X_te = df[fcols].values[s:]
    y_te = df["target_24h"].values[s:]
    
    print(f"   Features: {len(fcols)}")
    print(f"   Train: {df['datetime'].iloc[0].date()} → {df['datetime'].iloc[s-1].date()} ({len(X_tr):,} rows)")
    print(f"   Test : {df['datetime'].iloc[s].date()} → {df['datetime'].iloc[-1].date()} ({len(X_te):,} rows)")
    
    # Check for NaN in target
    if np.isnan(y_tr).any():
        nan_count = np.isnan(y_tr).sum()
        print(f"   WARNING: {nan_count} NaN values in training target")
    
    if np.isnan(y_te).any():
        nan_count = np.isnan(y_te).sum()
        print(f"   WARNING: {nan_count} NaN values in test target")
    
    return df, fcols, X_tr, y_tr, X_te, y_te, s


# ─────────────────────────────────────────────────────────────
# 2. CORRELATION FILTER
# ─────────────────────────────────────────────────────────────
def correlation_filter(df, fcols, split_idx, y_col="target_24h",
                       pearson_min=0.05, spearman_min=0.05,
                       inter_max=0.95):
    # FIX: restrict ALL computations to training rows only.
    # Original code used the full df which allowed test-set targets to
    # influence feature selection — a classic selection leakage.
    print(f"\n[2] Correlation filter (|r|>{pearson_min}, inter-r<{inter_max})...")
    print(f"   Using training rows only (rows 0:{split_idx}) to prevent selection leakage")

    df_train = df.iloc[:split_idx].copy()
    target = df_train[y_col]
    rows = []
    
    for c in fcols:
        valid = df_train[c].notna() & target.notna()
        if valid.sum() < 10:
            continue
        
        pear = float(df_train.loc[valid, c].corr(target[valid]))
        spear = float(scipy_stats.spearmanr(
            df_train.loc[valid, c].values,
            target[valid].values
        ).statistic)
        rows.append({
            "feature": c,
            "pearson": pear,
            "spearman": spear,
            "abs_pearson": abs(pear),
            "abs_spearman": abs(spear)
        })
    
    if not rows:
        print("   No features passed correlation check!")
        return [], pd.DataFrame()
    
    corr_df = pd.DataFrame(rows).sort_values("abs_pearson", ascending=False)
    
    # Step 1: minimum signal threshold
    keep_mask = (corr_df["abs_pearson"] >= pearson_min) | (corr_df["abs_spearman"] >= spearman_min)
    dropped_low = corr_df[~keep_mask]["feature"].tolist()
    corr_df = corr_df[keep_mask].reset_index(drop=True)
    print(f"   Dropped {len(dropped_low)} low-correlation features")
    
    if len(corr_df) == 0:
        print("   No features passed low-correlation threshold!")
        return [], pd.DataFrame()
    
    # Step 2: remove inter-feature collinearity (training rows only)
    surviving = corr_df["feature"].tolist()
    feat_corr = df_train[surviving].corr().abs()
    to_drop = set()
    
    for i, c1 in enumerate(surviving):
        if c1 in to_drop:
            continue
        for c2 in surviving[i+1:]:
            if c2 in to_drop:
                continue
            if feat_corr.loc[c1, c2] > inter_max:
                r1 = corr_df.loc[corr_df["feature"] == c1, "abs_pearson"].values[0]
                r2 = corr_df.loc[corr_df["feature"] == c2, "abs_pearson"].values[0]
                if r1 >= r2:
                    to_drop.add(c2)
                else:
                    to_drop.add(c1)
                    break
    
    corr_df = corr_df[~corr_df["feature"].isin(to_drop)].reset_index(drop=True)
    kept_cols = corr_df["feature"].tolist()
    print(f"   Removed {len(to_drop)} collinear features")
    print(f"   → {len(kept_cols)} features remain")
    
    # Plot (training data correlations only)
    if len(corr_df) > 0:
        top = corr_df.head(min(N_PLOT_TOP, len(corr_df)))
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        for ax, metric, label in [
            (axes[0], "pearson", "Pearson r"),
            (axes[1], "spearman", "Spearman ρ"),
        ]:
            vals = top[metric].values
            ax.barh(top["feature"][::-1], vals[::-1],
                    color=[GROUP_COLORS.get(get_group(c), "#90A4AE") for c in top["feature"][::-1]],
                    alpha=0.85)
            ax.axvline(0, color="black", lw=0.8)
            ax.axvline(pearson_min, color="red", ls="--", lw=1.2, label=f"Threshold {pearson_min}")
            ax.axvline(-pearson_min, color="red", ls="--", lw=1.2)
            ax.set_xlabel(f"{label} with Target (t+24)")
            ax.set_title(f"Top {len(top)} Features — {label}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis="x")
        
        from matplotlib.patches import Patch
        patches = [Patch(color=v, label=k) for k, v in GROUP_COLORS.items()]
        axes[1].legend(handles=patches, fontsize=7, loc="lower right", ncol=2)
        plt.suptitle("Figure S1. Correlation Filter — Features vs 24h-ahead Target",
                     fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "figS1_correlation_filter.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   → figS1_correlation_filter.png")
    
    return kept_cols, corr_df


# ─────────────────────────────────────────────────────────────
# 3. VIF FILTER (simplified)
# ─────────────────────────────────────────────────────────────
def vif_filter(df, fcols, split_idx, vif_max=10.0):
    # Evaluated on training rows only to prevent lookahead bias
    print(f"\n[3] VIF filter (threshold={vif_max}, training rows only)...")
    
    df_train = df.iloc[:split_idx]
    if len(fcols) < 3:
        print("   Too few features for VIF, skipping")
        return fcols, None
    
    from numpy.linalg import pinv
    
    cols = list(fcols)
    dropped = []
    
    # Pre-scan for near-zero variance (training data)
    for c in cols[:]:
        if df_train[c].std() < 1e-6:
            print(f"   Dropping constant feature: {c}")
            dropped.append((c, np.inf, "near-zero variance"))
            cols.remove(c)
    
    if len(cols) < 2:
        print("   Too few features after removing constants")
        return cols, None
    
    iteration = 0
    while len(cols) > 3 and iteration < 20:
        iteration += 1
        X = df_train[cols].values.astype(float)
        
        # Normalize
        X_norm = (X - X.mean(0)) / (X.std(0) + 1e-8)
        
        # Check for NaN/Inf
        if not np.isfinite(X_norm).all():
            print("   NaN/Inf detected, stopping VIF")
            break
        
        # Correlation matrix
        corr_mat = np.corrcoef(X_norm.T)
        corr_mat = (corr_mat + corr_mat.T) / 2.0
        np.fill_diagonal(corr_mat, 1.0)
        
        try:
            inv_corr = pinv(corr_mat)
            vifs = np.diag(inv_corr)
        except Exception as e:
            print(f"   VIF computation failed: {e}")
            break
        
        # Find max VIF
        valid_mask = (vifs >= 1.0) & np.isfinite(vifs)
        if not valid_mask.any():
            break
        
        vif_df = pd.DataFrame({"feature": cols, "VIF": vifs})
        vif_df = vif_df[vif_df["VIF"] >= 1.0].sort_values("VIF", ascending=False)
        
        if vif_df.empty:
            break
        
        max_vif = vif_df.iloc[0]["VIF"]
        max_col = vif_df.iloc[0]["feature"]
        
        if max_vif <= vif_max:
            break
        
        print(f"   Iter {iteration}: drop '{max_col}'  VIF={max_vif:.2f}")
        dropped.append((max_col, max_vif, "high VIF"))
        cols.remove(max_col)
    
    print(f"   Dropped {len(dropped)} high-VIF features")
    print(f"   → {len(cols)} features remain")
    
    return cols, None


# ─────────────────────────────────────────────────────────────
# 4. RANDOM FOREST IMPORTANCE
# ─────────────────────────────────────────────────────────────
def rf_importance(X_tr, y_tr, fcols):
    print("\n[4] Random Forest importance (MDI)...")
    
    if len(X_tr) == 0 or len(fcols) == 0:
        print("   No data for RF importance")
        return None, pd.DataFrame()
    
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=15, min_samples_leaf=3,
        max_features="sqrt", n_jobs=-1, random_state=RANDOM_SEED,
    )
    rf.fit(X_tr, y_tr)
    
    imp_df = pd.DataFrame({
        "feature": fcols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    
    top = imp_df.head(min(N_PLOT_TOP, len(imp_df)))
    if len(top) > 0:
        colors = [GROUP_COLORS.get(get_group(c), "#90A4AE") for c in top["feature"]]
        fig, ax = plt.subplots(figsize=(10, 9))
        ax.barh(top["feature"][::-1], top["importance"].values[::-1],
                color=colors[::-1], alpha=0.85)
        ax.set_xlabel("MDI Importance (mean decrease in impurity)")
        ax.set_title(f"Figure S3. Random Forest MDI Importance — Top {len(top)}")
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "figS3_rf_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   → figS3_rf_importance.png")
    
    print(f"   Top 5: {imp_df['feature'].head(5).tolist()}")
    return rf, imp_df


# ─────────────────────────────────────────────────────────────
# 5. PERMUTATION IMPORTANCE
# ─────────────────────────────────────────────────────────────
def perm_importance(rf, X_val, y_val, fcols):
    print("\n[5] Permutation importance (validation set)...")
    
    if rf is None or len(X_val) == 0:
        print("   No data for permutation importance")
        return pd.DataFrame()

    result = permutation_importance(
        rf, X_val, y_val,
        n_repeats=10, random_state=RANDOM_SEED, n_jobs=-1,
    )
    perm_df = pd.DataFrame({
        "feature": fcols,
        "perm_imp": result.importances_mean,
        "perm_std": result.importances_std,
    }).sort_values("perm_imp", ascending=False).reset_index(drop=True)
    
    top = perm_df.head(min(N_PLOT_TOP, len(perm_df)))
    if len(top) > 0:
        colors = [GROUP_COLORS.get(get_group(c), "#90A4AE") for c in top["feature"]]
        fig, ax = plt.subplots(figsize=(10, 9))
        ax.barh(top["feature"][::-1], top["perm_imp"].values[::-1],
                xerr=top["perm_std"].values[::-1],
                color=colors[::-1], alpha=0.85, capsize=3)
        ax.axvline(0, color="red", ls="--", lw=1.2, label="Zero (no effect)")
        ax.set_xlabel("Mean R² decrease when feature permuted")
        ax.set_title(f"Figure S4. Permutation Importance — Top {len(top)}")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "figS4_permutation.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   → figS4_permutation.png")
    
    print(f"   Top 5: {perm_df['feature'].head(5).tolist()}")
    return perm_df


# ─────────────────────────────────────────────────────────────
# 6. RFE
# ─────────────────────────────────────────────────────────────
def rfe_selection(X_tr, y_tr, X_val, y_val, fcols):
    print("\n[6] RFE — finding optimal feature count (scored on validation)...")

    if len(X_tr) == 0 or len(fcols) < 5:
        print("   Insufficient data for RFE")
        return fcols[:min(20, len(fcols))], len(fcols), 0

    n = len(fcols)
    eval_sizes = [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, n]
    eval_sizes = sorted(set([s for s in eval_sizes if s <= n]))

    rf_rfe = RandomForestRegressor(
        n_estimators=150, max_depth=12, min_samples_leaf=4,
        max_features="sqrt", n_jobs=-1, random_state=RANDOM_SEED,
    )

    r2_scores = []
    print(f"   Evaluating {len(eval_sizes)} subset sizes...")

    for size in eval_sizes:
        rfe = RFE(rf_rfe, n_features_to_select=size, step=max(1, size//10))
        rfe.fit(X_tr, y_tr)
        pred = rfe.predict(X_val)
        r2 = r2_score(y_val, np.clip(pred, 0, None))
        r2_scores.append(r2)
        print(f"     n={size:3d}: R²={r2:.4f}")
    
    max_r2 = max(r2_scores)
    best_idx = np.argmax(r2_scores)
    best_n = eval_sizes[best_idx]
    
    print(f"   Best n={best_n} with R²={max_r2:.4f}")
    
    # Final RFE
    rfe_final = RFE(rf_rfe, n_features_to_select=best_n, step=max(1, best_n//10))
    rfe_final.fit(X_tr, y_tr)
    rfe_cols = [fcols[i] for i, s in enumerate(rfe_final.support_) if s]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eval_sizes, r2_scores, "o-", color="#1565C0", lw=2, ms=5)
    ax.axvline(best_n, color="red", ls="--", lw=1.5, label=f"Best n={best_n}")
    ax.set_xlabel("Number of features")
    ax.set_ylabel("R² (validation set)")
    ax.set_title("Figure S5. RFE Curve — R² vs Feature Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "figS5_rfe_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   → figS5_rfe_curve.png")
    
    return rfe_cols, best_n, max_r2


# ─────────────────────────────────────────────────────────────
# 7. LASSO
# ─────────────────────────────────────────────────────────────
def lasso_selection(X_tr, y_tr, fcols):
    print("\n[7] LASSO CV for sparsity...")
    
    if len(X_tr) == 0 or len(fcols) == 0:
        print("   No data for LASSO")
        return []
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)

    # Use TimeSeriesSplit to ensure validation folds remain chronologically after training folds
    tscv = TimeSeriesSplit(n_splits=5)
    lasso = LassoCV(cv=tscv, n_alphas=100, max_iter=5000, random_state=RANDOM_SEED, n_jobs=-1)
    lasso.fit(X_tr_s, y_tr)
    
    lasso_mask = np.abs(lasso.coef_) > 1e-6
    lasso_cols = [fcols[i] for i, selected in enumerate(lasso_mask) if selected]
    
    print(f"   LASSO alpha={lasso.alpha_:.4f}")
    print(f"   Non-zero coefficients: {len(lasso_cols)} / {len(fcols)}")
    print(f"   Top LASSO features: {lasso_cols[:5]}")
    
    return lasso_cols


# ─────────────────────────────────────────────────────────────
# 8. VOTE AGGREGATION
# ─────────────────────────────────────────────────────────────
def aggregate_votes(fcols_all, corr_kept, vif_kept, imp_df, perm_df, rfe_cols, lasso_cols):
    print("\n[8] Vote aggregation...")
    
    corr_set = set(corr_kept)
    vif_set = set(vif_kept)
    imp_set = set(imp_df.head(40)["feature"]) if imp_df is not None and len(imp_df) > 0 else set()
    perm_set = set(perm_df.head(40)["feature"]) if perm_df is not None and len(perm_df) > 0 else set()
    rfe_set = set(rfe_cols)
    lasso_set = set(lasso_cols)
    
    rows = []
    for f in fcols_all:
        votes = sum([
            f in corr_set,
            f in vif_set,
            f in imp_set,
            f in perm_set,
            f in rfe_set,
            f in lasso_set,
        ])
        
        imp_val = imp_df.loc[imp_df["feature"] == f, "importance"].values if imp_df is not None else [0]
        perm_val = perm_df.loc[perm_df["feature"] == f, "perm_imp"].values if perm_df is not None else [0]
        
        rows.append({
            "feature": f,
            "votes": votes,
            "corr": int(f in corr_set),
            "vif": int(f in vif_set),
            "rf_imp": int(f in imp_set),
            "perm": int(f in perm_set),
            "rfe": int(f in rfe_set),
            "lasso": int(f in lasso_set),
            "mdi_imp": float(imp_val[0]) if len(imp_val) else 0.0,
            "perm_imp": float(perm_val[0]) if len(perm_val) else 0.0,
            "group": get_group(f),
        })
    
    vote_df = pd.DataFrame(rows).sort_values(["votes", "mdi_imp"], ascending=[False, False])
    
    # Select features with ≥3 votes
    final_set = vote_df[vote_df["votes"] >= 3]["feature"].tolist()
    
    if len(final_set) < 10:
        # Fallback: take top 20 by votes
        final_set = vote_df.head(20)["feature"].tolist()
    
    print(f"   Features with ≥3 votes: {len(final_set)}")
    print(f"   Final feature set: {len(final_set)}")
    
    return vote_df, final_set


# ─────────────────────────────────────────────────────────────
# 9. EVALUATE
# ─────────────────────────────────────────────────────────────
def evaluate_final(X_tr, y_tr, X_te, y_te, fcols_all, final_set):
    print("\n[9] Evaluating full vs final feature set...")
    
    rf_params = dict(
        n_estimators=300, max_depth=15, min_samples_leaf=3,
        max_features="sqrt", n_jobs=-1, random_state=RANDOM_SEED,
    )
    
    # Remove NaN from target
    train_mask = ~np.isnan(y_tr)
    test_mask = ~np.isnan(y_te)
    
    # Full
    rf_full = RandomForestRegressor(**rf_params)
    rf_full.fit(X_tr[train_mask], y_tr[train_mask])
    pred_full = rf_full.predict(X_te[test_mask])
    r2_full = r2_score(y_te[test_mask], np.clip(pred_full, 0, None))
    
    # Final subset
    final_idx = [fcols_all.index(f) for f in final_set if f in fcols_all]
    if len(final_idx) == 0:
        print("   No final features selected!")
        return r2_full, 0
    
    rf_final = RandomForestRegressor(**rf_params)
    rf_final.fit(X_tr[train_mask][:, final_idx], y_tr[train_mask])
    pred_final = rf_final.predict(X_te[test_mask][:, final_idx])
    r2_final = r2_score(y_te[test_mask], np.clip(pred_final, 0, None))
    
    mae_full = mean_absolute_error(y_te[test_mask], np.clip(pred_full, 0, None))
    mae_final = mean_absolute_error(y_te[test_mask], np.clip(pred_final, 0, None))
    
    print(f"   All features ({len(fcols_all)}): R²={r2_full:.4f}  MAE={mae_full:.3f}")
    print(f"   Final set   ({len(final_set)}):   R²={r2_final:.4f}  MAE={mae_final:.3f}")
    
    return r2_full, r2_final


# ─────────────────────────────────────────────────────────────
# 10. FINAL PLOT
# ─────────────────────────────────────────────────────────────
def plot_final(vote_df, final_set):
    top = vote_df.head(N_PLOT_TOP).copy()
    top["selected"] = top["feature"].isin(final_set)
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], figure=fig)
    
    # Left: vote scores
    ax1 = fig.add_subplot(gs[0])
    colors = [GROUP_COLORS.get(get_group(c), "#90A4AE") for c in top["feature"]]
    edge_colors = ["black" if s else "none" for s in top["selected"]]
    ax1.barh(top["feature"][::-1], top["votes"].values[::-1],
             color=colors[::-1], edgecolor=edge_colors[::-1],
             linewidth=1.2, alpha=0.85)
    ax1.axvline(3, color="red", ls="--", lw=1.5, label="3-vote threshold")
    ax1.set_xlabel("Number of methods selecting this feature (max 6)")
    ax1.set_title(f"Figure S6. Feature Votes — Top {len(top)}")
    ax1.set_xlim(0, 6.5)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="x")
    
    # Right: group breakdown
    ax2 = fig.add_subplot(gs[1])
    final_df = vote_df[vote_df["feature"].isin(final_set)].copy()
    if len(final_df) > 0:
        grp_counts = final_df.groupby("group").size().sort_values(ascending=True)
        grp_colors = [GROUP_COLORS.get(g, "#90A4AE") for g in grp_counts.index]
        ax2.barh(grp_counts.index, grp_counts.values, color=grp_colors, alpha=0.85)
        ax2.set_xlabel("Feature count")
        ax2.set_title(f"Final Set ({len(final_set)} features)\nby group")
        ax2.grid(True, alpha=0.3, axis="x")
    
    plt.suptitle("Feature Selection Summary — Multi-Method Vote (6 methods)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "figS6_final_set.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   → figS6_final_set.png")


# ─────────────────────────────────────────────────────────────
# 11. SAVE REPORT
# ─────────────────────────────────────────────────────────────
def save_report(vote_df, final_set, r2_full, r2_final, n_all):
    lines = [
        "=" * 65,
        "  FEATURE SELECTION REPORT — 24h Building Energy Forecast",
        "=" * 65,
        "",
        f"  Total features input  : {n_all}",
        f"  Final feature set     : {len(final_set)}",
        f"  R² (all features)     : {r2_full:.4f}",
        f"  R² (final set)        : {r2_final:.4f}",
        "",
        "  FINAL FEATURE SET",
        "  ─" * 30,
    ]
    
    for i, f in enumerate(final_set, 1):
        lines.append(f"  {i:2d}. {f}")
    
    report = "\n".join(lines)
    REPORT_TXT.write_text(report, encoding="utf-8")
    FINAL_LIST_TXT.write_text("\n".join(final_set), encoding="utf-8")

    print(f"\n  Saved:")
    print(f"    {REPORT_TXT.name}")
    print(f"    {FINAL_LIST_TXT.name}")


# ─────────────────────────────────────────────────────────────
# 12. SAVE METADATA & FINAL DATASET
# ─────────────────────────────────────────────────────────────
def save_selection_json(vote_df, final_set, n_all, r2_full, r2_final):
    votes_table = [
        {
            "feature": r["feature"],
            "votes": int(r["votes"]),
            "corr": int(r["corr"]),
            "vif": int(r["vif"]),
            "rf_imp": int(r["rf_imp"]),
            "perm": int(r["perm"]),
            "rfe": int(r["rfe"]),
            "lasso": int(r["lasso"]),
            "mdi_imp": float(r["mdi_imp"]),
            "perm_imp": float(r["perm_imp"]),
            "group": r["group"],
        }
        for r in vote_df.to_dict(orient="records")
    ]

    data = {
        "n_input": int(n_all),
        "n_final": int(len(final_set)),
        "r2_all_features": float(r2_full),
        "r2_final_features": float(r2_final),
        "final_features": list(final_set),
        "votes_table": votes_table,
    }
    SELECTION_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"    {SELECTION_JSON.name}")


def save_final_data_csv(df, final_set):
    meta_cols = [c for c in ("datetime", "sn") if c in df.columns]
    keep_cols = meta_cols + [c for c in final_set if c in df.columns] + ["target_24h"]
    df[keep_cols].to_csv(FINAL_FEAT_CSV, index=False)
    print(f"    {FINAL_FEAT_CSV.name}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  FEATURE SELECTION — 24h Building Energy Forecast")
    print("=" * 65)
    
    try:
        # 1. Load
        df, fcols, X_tr, y_tr, X_te, y_te, split_idx = load_features()
        n_all = len(fcols)

        # Embargo: purge the last EMBARGO_H rows of training data to prevent label overlap
        X_tr = X_tr[:-EMBARGO_H]
        y_tr = y_tr[:-EMBARGO_H]
        split_purged = split_idx - EMBARGO_H
        print(f"   Embargo: purged last {EMBARGO_H} train rows "
              f"(train now 0:{split_purged}; test start unchanged at {split_idx})")

        # Remove NaN from targets
        train_mask = ~np.isnan(y_tr)
        test_mask = ~np.isnan(y_te)
        
        X_tr = X_tr[train_mask]
        y_tr = y_tr[train_mask]
        X_te = X_te[test_mask]
        y_te = y_te[test_mask]
        
        # 2. Correlation filter (training data only)
        corr_kept, corr_df = correlation_filter(df, fcols, split_purged)
        
        if len(corr_kept) == 0:
            print("ERROR: No features passed correlation filter!")
            return
        
        # 3. VIF filter (training data only)
        vif_kept, vif_df = vif_filter(df, corr_kept, split_purged)

        # Prepare FIT / VAL split from training data (with boundary embargo)
        fit_end = int(split_purged * (1 - VAL_RATIO))
        fit_purged = fit_end - EMBARGO_H
        print(f"\n   FIT/VAL split within train: fit=0:{fit_purged} "
              f"(embargo {EMBARGO_H}h)  val={fit_end}:{split_purged}")

        full_train_mask = ~np.isnan(df["target_24h"].values[:split_purged])

        X_fit_vif = df[vif_kept].values[:fit_purged]
        X_fit_vif = X_fit_vif[full_train_mask[:fit_purged]]
        y_fit = df["target_24h"].values[:fit_purged]
        y_fit = y_fit[full_train_mask[:fit_purged]]

        X_val_vif = df[vif_kept].values[fit_end:split_purged]
        X_val_vif = X_val_vif[full_train_mask[fit_end:split_purged]]
        y_val = df["target_24h"].values[fit_end:split_purged]
        y_val = y_val[full_train_mask[fit_end:split_purged]]

        # 4. Random Forest feature importance (fit split)
        rf_model, imp_df = rf_importance(X_fit_vif, y_fit, vif_kept)

        # 5. Permutation importance (validation split)
        perm_df = perm_importance(rf_model, X_val_vif, y_val, vif_kept)

        # 6. Recursive Feature Elimination (evaluated on validation split)
        rfe_cols, best_n, max_r2 = rfe_selection(X_fit_vif, y_fit, X_val_vif, y_val, vif_kept)

        # 7. LASSO regression sparsity (fit split)
        lasso_cols = lasso_selection(X_fit_vif, y_fit, vif_kept)
        
        # 8. Vote aggregation
        vote_df, final_set = aggregate_votes(
            fcols, corr_kept, vif_kept, imp_df, perm_df, rfe_cols, lasso_cols
        )
        
        # 9. Evaluate
        r2_full, r2_final = evaluate_final(X_tr, y_tr, X_te, y_te, fcols, final_set)
        
        # 10. Final plot
        plot_final(vote_df, final_set)
        
        # 11. Save
        save_report(vote_df, final_set, r2_full, r2_final, n_all)
        save_selection_json(vote_df, final_set, n_all, r2_full, r2_final)
        save_final_data_csv(df, final_set)

        print(f"\n{'=' * 65}")
        print(f"  DONE — {len(final_set)} features selected from {n_all}")
        print(f"  R²: {r2_full:.4f} → {r2_final:.4f}")
        print(f"  Outputs → {OUT_DIR}/")
        print(f"{'=' * 65}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()