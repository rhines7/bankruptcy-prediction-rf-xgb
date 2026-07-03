"""Random Forest baseline, hyperparameter tuning, and cross-validated evaluation."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_val_predict

from bankruptcy.config import (
    RF_CV_FOLDS,
    RF_CV_SEED,
    RF_N_ESTIMATORS,
    RF_PARAM_GRID,
    RF_PARAM_GRID_QUICK,
)


def _make_cv():
    return StratifiedKFold(n_splits=RF_CV_FOLDS, shuffle=True, random_state=RF_CV_SEED)


def _make_rf(**params):
    return RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RF_CV_SEED,
        n_jobs=-1,
        class_weight="balanced",
        **params,
    )


def evaluate_rf_baseline(X_train, y_train, X_test, y_test):
    """Out-of-fold CV on train and test-set ROC-AUC for the default RF."""
    cv = _make_cv()
    rf = _make_rf()

    y_train_proba_cv = cross_val_predict(
        rf, X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]
    cv_auc = roc_auc_score(y_train, y_train_proba_cv)

    rf.fit(X_train, y_train)
    y_test_proba = rf.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_proba)

    return {
        "model": rf,
        "cv_auc": cv_auc,
        "test_auc": test_auc,
        "y_test_proba": y_test_proba,
        "y_train_proba_cv": y_train_proba_cv,
    }


def tune_random_forest(X_train, y_train, quick=False, verbose=False):
    """
    Grid search over four RF hyperparameters (tree count fixed).

    Returns tuning results, best params, and a classifier instance (unfitted).
    """
    param_grid = RF_PARAM_GRID_QUICK if quick else RF_PARAM_GRID
    param_list = list(ParameterGrid(param_grid))
    cv = _make_cv()

    print(f"Running Random Forest hyperparameter search over {len(param_list)} configurations...")

    tuning_results = []
    best_cv_auc = -np.inf
    best_params = None

    for config_rank, params in enumerate(param_list, start=1):
        if verbose:
            print(f"Config {config_rank}/{len(param_list)}: {params}")

        fold_aucs = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
            rf = _make_rf(**params)
            rf.fit(X_train[train_idx], y_train[train_idx])
            y_val_proba = rf.predict_proba(X_train[val_idx])[:, 1]
            fold_auc = roc_auc_score(y_train[val_idx], y_val_proba)
            fold_aucs.append(fold_auc)
            if verbose:
                print(f"  Fold {fold_idx} AUC: {fold_auc:.4f}")

        mean_auc = float(np.mean(fold_aucs))
        std_auc = float(np.std(fold_aucs))
        if verbose:
            print(f"  Mean CV AUC: {mean_auc:.4f} (+/- {std_auc:.4f})\n")

        tuning_results.append(
            {"params": params, "mean_cv_roc_auc": mean_auc, "std_cv_roc_auc": std_auc}
        )
        if mean_auc > best_cv_auc:
            best_cv_auc = mean_auc
            best_params = params

    print("Best RF parameters:", best_params)
    print(f"Best tuning CV ROC-AUC: {best_cv_auc:.4f}")

    best_rf = _make_rf(**best_params)
    return {
        "tuning_results": tuning_results,
        "best_params": best_params,
        "best_tuning_cv_auc": best_cv_auc,
        "model": best_rf,
    }


def evaluate_rf_tuned(best_rf, X_train, y_train, X_test, y_test):
    """OOF CV and test evaluation for a tuned RF configuration."""
    cv = _make_cv()

    y_train_proba_cv = cross_val_predict(
        best_rf, X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]
    cv_auc = roc_auc_score(y_train, y_train_proba_cv)

    best_rf.fit(X_train, y_train)
    y_test_proba = best_rf.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_proba)

    return {
        "cv_auc": cv_auc,
        "test_auc": test_auc,
        "y_test_proba": y_test_proba,
        "y_train_proba_cv": y_train_proba_cv,
        "model": best_rf,
    }


def top_rf_configurations(tuning_results, n=5):
    """Format top-N RF tuning rows for reporting."""
    sorted_results = sorted(
        tuning_results, key=lambda r: r["mean_cv_roc_auc"], reverse=True
    )[:n]
    rows = []
    for rank, row in enumerate(sorted_results, start=1):
        params = row["params"]
        rows.append(
            {
                "rank": rank,
                "mean_cv_roc_auc": row["mean_cv_roc_auc"],
                "std_cv_roc_auc": row["std_cv_roc_auc"],
                "max_depth": params["max_depth"],
                "min_samples_split": params["min_samples_split"],
                "min_samples_leaf": params["min_samples_leaf"],
                "max_features": params["max_features"],
            }
        )
    return rows
