"""Load ARFF bankruptcy data and print dataset diagnostics."""

import numpy as np
from scipy.io import arff

from bankruptcy.config import FIVE_YEAR_FILE, FOUR_YEAR_FILE


def load_arff_as_numpy(path):
    """Load an ARFF file into feature matrix X and binary label vector y."""
    data, meta = arff.loadarff(path)
    feature_names = meta.names()[:-1]
    class_name = meta.names()[-1]

    num_attributes = len(meta.names())

    X = np.zeros((data.shape[0], num_attributes - 1), dtype=float)
    for j, name in enumerate(feature_names):
        X[:, j] = data[name].astype(float)

    raw_y = data[class_name]
    y = np.array([int(v) for v in raw_y], dtype=int)
    return X, y, feature_names, class_name


def load_temporal_datasets():
    """Load 4th-year (train) and 5th-year (test) datasets."""
    X_train, y_train, feature_names, class_name = load_arff_as_numpy(FOUR_YEAR_FILE)
    X_test, y_test, _, _ = load_arff_as_numpy(FIVE_YEAR_FILE)
    return X_train, y_train, X_test, y_test, feature_names, class_name


def summarize_dataset(X, y, label):
    """Print shape, class balance, and missing-value counts for one cohort."""
    print(f"Class distribution ({label}):")
    n = y.shape[0]
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    print(f"  Non-bankrupt (0): {n0} ({100 * n0 / n:.2f}%)")
    print(f"  Bankrupt (1):     {n1} ({100 * n1 / n:.2f}%)")
    missing = int(np.isnan(X).sum())
    print(f"  Total missing entries in X: {missing}\n")


def explore_datasets(X_train, y_train, X_test, y_test, feature_names):
    """Print diagnostics for train and test cohorts."""
    print(
        f"4-year dataset (train): {X_train.shape[0]} samples, {X_train.shape[1]} features"
    )
    print(
        f"5-year dataset (test):  {X_test.shape[0]} samples, {X_test.shape[1]} features"
    )
    summarize_dataset(X_train, y_train, "4th year (train)")
    summarize_dataset(X_test, y_test, "5th year (test)")

    missing_counts = np.isnan(X_train).sum(axis=0)
    print("Missing values per feature (first 10, from train):")
    for idx, (name, count) in enumerate(zip(feature_names, missing_counts)):
        if idx >= 10:
            break
        print(f"  {name}: {count} missing")
