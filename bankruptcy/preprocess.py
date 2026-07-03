"""Temporal split preprocessing: imputation and scaling fit on train only."""

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def prepare_features(X_train_raw, y_train, X_test_raw, y_test):
    """
    Apply median imputation and standardization.

    Fitting on the 4th-year training data only prevents leakage into the
    5th-year hold-out cohort.
    """
    print("Train (4th year) shape:", X_train_raw.shape, "Test (5th year) shape:", X_test_raw.shape)
    print("Train class balance:")
    print(
        f"  Non-bankrupt (0): {(y_train == 0).sum()} / {y_train.shape[0]}",
        f"({(y_train == 0).mean():.2%})",
    )
    print(
        f"  Bankrupt (1):     {(y_train == 1).sum()} / {y_train.shape[0]}",
        f"({(y_train == 1).mean():.2%})",
    )
    print("\nTest class balance:")
    print(
        f"  Non-bankrupt (0): {(y_test == 0).sum()} / {y_test.shape[0]}",
        f"({(y_test == 0).mean():.2%})",
    )
    print(
        f"  Bankrupt (1):     {(y_test == 1).sum()} / {y_test.shape[0]}",
        f"({(y_test == 1).mean():.2%})",
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train_raw)
    X_test_imputed = imputer.transform(X_test_raw)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_imputed)
    X_test = scaler.transform(X_test_imputed)

    print("\nAfter imputation and scaling:")
    print("  X_train shape:", X_train.shape)
    print("  X_test shape:", X_test.shape)

    return X_train, X_test, imputer, scaler
