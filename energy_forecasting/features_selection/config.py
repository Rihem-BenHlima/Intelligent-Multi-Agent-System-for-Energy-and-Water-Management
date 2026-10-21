"""
config.py
=========
Path and file configuration for feature selection.
"""

from pathlib import Path

# --- Root Directory --------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

print(f"Config: ROOT = {ROOT}")

# --- Input Files -----------------------------------------------------------
FEATURES_CSV = ROOT / "features_engineering" / "output" / "features.csv"

# --- Output Directory & Files ----------------------------------------------
OUT_DIR = ROOT / "features_selection" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_TXT       = OUT_DIR / "feature_selection_report.txt"
FINAL_LIST_TXT   = OUT_DIR / "final_feature_set.txt"
SELECTION_JSON   = OUT_DIR / "feature_selection.json"
FINAL_FEAT_CSV   = OUT_DIR / "features_final_data.csv"

# --- Plot Files ------------------------------------------------------------
FIG_CORR    = OUT_DIR / "figS1_correlation_filter.png"
FIG_VIF     = OUT_DIR / "figS2_vif.png"
FIG_RF      = OUT_DIR / "figS3_rf_importance.png"
FIG_PERM    = OUT_DIR / "figS4_permutation.png"
FIG_RFE     = OUT_DIR / "figS5_rfe_curve.png"
FIG_FINAL   = OUT_DIR / "figS6_final_set.png"

# ── Validation ────────────────────────────────────────────────────────────
def validate_paths():
    """Print path information for debugging"""
    print("\n" + "=" * 50)
    print("  CONFIGURATION PATHS")
    print("=" * 50)
    print(f"  ROOT:           {ROOT}")
    print(f"  FEATURES_CSV:   {FEATURES_CSV}")
    print(f"    Exists:       {FEATURES_CSV.exists()}")
    print(f"  OUT_DIR:        {OUT_DIR}")
    print(f"    Exists:       {OUT_DIR.exists()}")
    print("=" * 50 + "\n")
    
    if not FEATURES_CSV.exists():
        print(f"  WARNING: Features file not found at {FEATURES_CSV}")
        print(f"  Please run features_engineering first to generate features.csv")

if __name__ == "__main__":
    validate_paths()