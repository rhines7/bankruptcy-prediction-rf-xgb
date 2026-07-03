"""
Bankruptcy prediction using Random Forest and XGBoost.

Entry point: python bankruptcy_classifier.py
Implementation: bankruptcy/ package.
"""

from bankruptcy.config import DATA_DIR, FIGURES_DIR, PROJECT_ROOT
from bankruptcy.io import explore_datasets, load_arff_as_numpy, load_temporal_datasets
from bankruptcy.pipeline import main
from bankruptcy.preprocess import prepare_features
from bankruptcy.rf_model import (
    evaluate_rf_baseline,
    evaluate_rf_tuned,
    top_rf_configurations,
    tune_random_forest,
)
from bankruptcy.viz import plot_confusion_matrix, plot_roc_comparison, save_figures
from bankruptcy.xgb_model import (
    evaluate_xgb_baseline,
    fit_xgb_tuned,
    top_xgb_configurations,
    tune_xgboost,
    xgb_oof_predictions,
)

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "FIGURES_DIR",
    "load_arff_as_numpy",
    "load_temporal_datasets",
    "explore_datasets",
    "prepare_features",
    "evaluate_rf_baseline",
    "tune_random_forest",
    "evaluate_rf_tuned",
    "top_rf_configurations",
    "evaluate_xgb_baseline",
    "tune_xgboost",
    "fit_xgb_tuned",
    "xgb_oof_predictions",
    "top_xgb_configurations",
    "plot_roc_comparison",
    "plot_confusion_matrix",
    "save_figures",
    "main",
]

if __name__ == "__main__":
    main()
