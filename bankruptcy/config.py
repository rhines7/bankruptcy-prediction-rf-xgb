"""Project paths, reproducibility settings, and hyperparameter grids."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"

FOUR_YEAR_FILE = DATA_DIR / "4year.arff"
FIVE_YEAR_FILE = DATA_DIR / "5year.arff"

# Fixed model complexity — not searched during tuning.
RF_N_ESTIMATORS = 300
XGB_N_ESTIMATORS = 1000
EARLY_STOPPING_ROUNDS = 50
CLASSIFICATION_THRESHOLD = 0.5

RF_CV_SEED = 42
XGB_TUNING_CV_SEED = 42
XGB_OOF_CV_SEED = 123
XGB_BASELINE_SPLIT_SEED = 42
XGB_TUNED_SPLIT_SEED = 123
XGB_SAMPLE_SEED = 42

RF_CV_FOLDS = 5
XGB_TUNING_CV_FOLDS = 3
XGB_OOF_CV_FOLDS = 5
XGB_TUNING_SAMPLES = 40
XGB_TUNING_SAMPLES_QUICK = 5

RF_PARAM_GRID = {
    "max_depth": [None, 5, 10, 20, 40],
    "min_samples_split": [2, 5, 10, 20, 40],
    "min_samples_leaf": [1, 2, 4, 8, 16],
    "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
}

# Smaller grid for smoke tests; centered on the best full-run configuration.
RF_PARAM_GRID_QUICK = {
    "max_depth": [None, 40],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [4],
    "max_features": [0.7],
}

XGB_PARAM_GRID = {
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1],
    "min_child_weight": [1, 3, 5, 7, 10],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_lambda": [0.0, 0.5, 1.0, 1.5, 2.0],
}

XGB_BASELINE_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}
