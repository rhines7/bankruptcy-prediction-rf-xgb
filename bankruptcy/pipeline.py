"""End-to-end pipeline orchestration."""

import argparse
import time

from bankruptcy.config import FIGURES_DIR
from bankruptcy.io import explore_datasets, load_temporal_datasets
from bankruptcy.preprocess import prepare_features
from bankruptcy.rf_model import (
    evaluate_rf_baseline,
    evaluate_rf_tuned,
    top_rf_configurations,
    tune_random_forest,
)
from bankruptcy.viz import save_figures
from bankruptcy.xgb_model import (
    evaluate_xgb_baseline,
    fit_xgb_tuned,
    top_xgb_configurations,
    tune_xgboost,
    xgb_oof_predictions,
)


def _print_summary(metrics):
    print("\n--- Summary ---", flush=True)
    for label, value in metrics.items():
        print(f"{label}: {value:.4f}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Bankruptcy prediction with Random Forest and XGBoost."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use reduced hyperparameter search for smoke testing.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-configuration and per-fold tuning progress.",
    )
    args = parser.parse_args(argv)

    start_time = time.time()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Figures: {FIGURES_DIR.resolve()}\n", flush=True)

    print("--- Load data ---", flush=True)
    X_train_raw, y_train, X_test_raw, y_test, feature_names, _ = load_temporal_datasets()
    explore_datasets(X_train_raw, y_train, X_test_raw, y_test, feature_names)

    print("\n--- Preprocess ---")
    X_train, X_test, _, _ = prepare_features(
        X_train_raw, y_train, X_test_raw, y_test
    )

    print("\n--- Random Forest ---")
    rf_baseline = evaluate_rf_baseline(X_train, y_train, X_test, y_test)
    print(f"Baseline RF OOF CV ROC-AUC: {rf_baseline['cv_auc']:.4f}")
    print(f"Baseline RF test ROC-AUC: {rf_baseline['test_auc']:.4f}")

    rf_tune = tune_random_forest(
        X_train, y_train, quick=args.quick, verbose=args.verbose
    )
    rf_tuned = evaluate_rf_tuned(
        rf_tune["model"], X_train, y_train, X_test, y_test
    )
    print(f"Tuned RF OOF CV ROC-AUC: {rf_tuned['cv_auc']:.4f}")
    print(f"Tuned RF test ROC-AUC: {rf_tuned['test_auc']:.4f}")

    rf_top5 = top_rf_configurations(rf_tune["tuning_results"])
    print("\nTop 5 RF configurations:")
    for row in rf_top5:
        print(row)

    print("\n--- XGBoost ---")
    xgb_baseline = evaluate_xgb_baseline(
        X_train, y_train, X_test, y_test, verbose=args.verbose
    )

    xgb_tune = tune_xgboost(X_train, y_train, quick=args.quick, verbose=args.verbose)
    xgb_tuned = fit_xgb_tuned(
        X_train,
        y_train,
        X_test,
        y_test,
        xgb_tune["best_params"],
        verbose=args.verbose,
    )
    xgb_oof = xgb_oof_predictions(
        X_train, y_train, xgb_tune["best_params"], verbose=args.verbose
    )

    xgb_top5 = top_xgb_configurations(xgb_tune["tuning_results"])
    print("\nTop 5 XGBoost configurations:")
    for row in xgb_top5:
        print(row)

    print("\n--- Figures ---")
    save_figures(
        y_test,
        rf_baseline["y_test_proba"],
        rf_tuned["y_test_proba"],
        xgb_tuned["y_test_proba"],
    )

    metrics = {
        "RF baseline OOF CV AUC": rf_baseline["cv_auc"],
        "RF baseline test AUC": rf_baseline["test_auc"],
        "RF tuned tuning CV mean": rf_tune["best_tuning_cv_auc"],
        "RF tuned OOF CV AUC": rf_tuned["cv_auc"],
        "RF tuned test AUC": rf_tuned["test_auc"],
        "XGB baseline validation AUC": xgb_baseline["val_auc"],
        "XGB baseline test AUC": xgb_baseline["test_auc"],
        "XGB tuned tuning CV mean (3-fold)": xgb_tune["best_tuning_cv_auc"],
        "XGB tuned validation AUC": xgb_tuned["val_auc"],
        "XGB tuned OOF CV AUC (5-fold)": xgb_oof["cv_auc"],
        "XGB tuned test AUC": xgb_tuned["test_auc"],
    }
    _print_summary(metrics)

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed / 60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
