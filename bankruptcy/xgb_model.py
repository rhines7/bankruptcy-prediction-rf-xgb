"""XGBoost baseline, hyperparameter tuning, early stopping, and OOF CV."""

from itertools import product

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from bankruptcy.config import (
    EARLY_STOPPING_ROUNDS,
    XGB_BASELINE_PARAMS,
    XGB_BASELINE_SPLIT_SEED,
    XGB_N_ESTIMATORS,
    XGB_OOF_CV_FOLDS,
    XGB_OOF_CV_SEED,
    XGB_PARAM_GRID,
    XGB_SAMPLE_SEED,
    XGB_TUNED_SPLIT_SEED,
    XGB_TUNING_CV_FOLDS,
    XGB_TUNING_CV_SEED,
    XGB_TUNING_SAMPLES,
    XGB_TUNING_SAMPLES_QUICK,
)


def _make_xgb(**params):
    return XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
        random_state=XGB_TUNING_CV_SEED,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        **params,
    )


def evaluate_xgb_baseline(X_train, y_train, X_test, y_test, verbose=False):
    """Train baseline XGB with early stopping on an internal validation split."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=XGB_BASELINE_SPLIT_SEED,
    )

    print("XGBoost baseline: training with early stopping...")
    print("  Training set:", X_tr.shape, "Validation set:", X_val.shape)

    model = _make_xgb(**XGB_BASELINE_PARAMS)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=50 if verbose else False,
    )

    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    y_test_proba = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_proba)

    print(f"Baseline XGBoost validation ROC-AUC: {val_auc:.4f}")
    print(f"Baseline XGBoost test ROC-AUC: {test_auc:.4f}")

    return {
        "model": model,
        "val_auc": val_auc,
        "test_auc": test_auc,
        "y_test_proba": y_test_proba,
    }


def tune_xgboost(X_train, y_train, quick=False, verbose=False):
    """Sample hyperparameter configurations; early stopping inside each CV fold."""
    param_names = list(XGB_PARAM_GRID.keys())
    param_values = [XGB_PARAM_GRID[name] for name in param_names]
    all_combinations = list(product(*param_values))
    num_total = len(all_combinations)

    num_candidates = XGB_TUNING_SAMPLES_QUICK if quick else XGB_TUNING_SAMPLES
    rng = np.random.default_rng(XGB_SAMPLE_SEED)
    selected_indices = rng.choice(num_total, size=num_candidates, replace=False)

    print(f"Total XGBoost grid size: {num_total} combinations")
    print(f"Sampling {num_candidates} configurations for tuning...")

    xgb_cv = StratifiedKFold(
        n_splits=XGB_TUNING_CV_FOLDS, shuffle=True, random_state=XGB_TUNING_CV_SEED
    )

    tuning_results = []
    best_params = None
    best_cv_auc = -np.inf

    for config_rank, combo_idx in enumerate(selected_indices, start=1):
        params = dict(zip(param_names, all_combinations[combo_idx]))

        if verbose:
            print(f"Config {config_rank}/{num_candidates}: {params}")

        fold_aucs = []
        for train_idx, val_idx in xgb_cv.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model = _make_xgb(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

            fold_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
            fold_aucs.append(fold_auc)

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

    print("Best XGBoost parameters:", best_params)
    print(f"Best tuning CV ROC-AUC: {best_cv_auc:.4f}")

    return {
        "tuning_results": tuning_results,
        "best_params": best_params,
        "best_tuning_cv_auc": best_cv_auc,
    }


def fit_xgb_tuned(X_train, y_train, X_test, y_test, best_params, verbose=False):
    """Fit tuned XGB on a fresh train/validation split; evaluate on test."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=XGB_TUNED_SPLIT_SEED,
    )

    print("Training tuned XGBoost with early stopping...")
    print("  Training set:", X_tr.shape, "Validation set:", X_val.shape)

    model = _make_xgb(**best_params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=50 if verbose else False,
    )

    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    y_test_proba = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_proba)

    print(f"Tuned XGBoost validation ROC-AUC: {val_auc:.4f}")
    print(f"Tuned XGBoost test ROC-AUC: {test_auc:.4f}")

    return {
        "model": model,
        "val_auc": val_auc,
        "test_auc": test_auc,
        "y_test_proba": y_test_proba,
    }


def xgb_oof_predictions(X_train, y_train, best_params, verbose=False):
    """5-fold OOF probabilities for the tuned XGB configuration."""
    cv_xgb = StratifiedKFold(
        n_splits=XGB_OOF_CV_FOLDS, shuffle=True, random_state=XGB_OOF_CV_SEED
    )

    print("Generating cross-validated predictions for tuned XGBoost...")
    y_train_proba_cv = np.zeros_like(y_train, dtype=float)

    for fold_idx, (train_idx, val_idx) in enumerate(cv_xgb.split(X_train, y_train), start=1):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        if verbose:
            print(f"  Fold {fold_idx}: training tuned XGBoost with early stopping...")

        model = XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            n_jobs=-1,
            random_state=42 + fold_idx,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            **best_params,
        )
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            verbose=50 if verbose else False,
        )

        y_val_proba = model.predict_proba(X_val)[:, 1]
        y_train_proba_cv[val_idx] = y_val_proba

        if verbose:
            fold_auc = roc_auc_score(y_val, y_val_proba)
            print(f"  Fold {fold_idx} ROC-AUC: {fold_auc:.4f}\n")

    cv_auc = roc_auc_score(y_train, y_train_proba_cv)
    print(f"Tuned XGBoost OOF CV ROC-AUC (training folds): {cv_auc:.4f}")

    return {"cv_auc": cv_auc, "y_train_proba_cv": y_train_proba_cv}


def top_xgb_configurations(tuning_results, n=5):
    """Format top-N XGB tuning rows for reporting."""
    sorted_results = sorted(
        tuning_results, key=lambda r: r["mean_cv_roc_auc"], reverse=True
    )[:n]
    rows = []
    for rank, result in enumerate(sorted_results, start=1):
        row = {
            "rank": rank,
            "mean_cv_roc_auc": float(result["mean_cv_roc_auc"]),
            "std_cv_roc_auc": float(result["std_cv_roc_auc"]),
        }
        row.update(result["params"])
        rows.append(row)
    return rows
