
"""
Dataset-specific feature engineering for digital payment fraud detection.

Credit Card:
    Preserves the existing numeric feature set.

PaySim:
    Retains transaction and balance features.
    Removes customer identifiers and the rule-based isFlaggedFraud field.

E-commerce:
    Derives account age and purchase-time features.
    Retains useful categorical and numeric features.
    Removes raw identifiers, IP addresses, and raw date strings.

This module does not fit encoders or scalers. Those must be fitted
inside the training pipeline to prevent data leakage.
"""

from typing import Literal

import pandas as pd


DatasetId = Literal["credit_card", "paysim", "ecommerce"]


def engineer_features(
    df: pd.DataFrame,
    dataset_id: DatasetId,
) -> pd.DataFrame:
    """
    Apply dataset-specific feature engineering to a feature-only DataFrame.

    The target column must not be included in df.

    Args:
        df: Input feature DataFrame.
        dataset_id: One of credit_card, paysim, ecommerce.

    Returns:
        A new DataFrame containing engineered features.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    X = df.copy()

    if dataset_id == "credit_card":
        # Preserve the established Credit Card feature set.
        return X

    if dataset_id == "paysim":
        required = {"type"}
        missing = required - set(X.columns)

        if missing:
            raise ValueError(
                f"PaySim dataset is missing required columns: {sorted(missing)}"
            )

        # Customer identifiers are not used as predictive features.
        X = X.drop(
            columns=["nameOrig", "nameDest", "isFlaggedFraud"],
            errors="ignore",
        )

        X["type"] = X["type"].fillna("Unknown").astype(str)

        return X

    if dataset_id == "ecommerce":
        required = {"signup_time", "purchase_time"}
        missing = required - set(X.columns)

        if missing:
            raise ValueError(
                f"E-commerce dataset is missing required columns: {sorted(missing)}"
            )

        signup = pd.to_datetime(
            X["signup_time"], errors="coerce"
        )
        purchase = pd.to_datetime(
            X["purchase_time"], errors="coerce"
        )

        # Time between account registration and purchase, in hours.
        X["account_age_hours"] = (
            purchase - signup
        ).dt.total_seconds() / 3600.0

        # Purchase-time features.
        X["purchase_hour"] = purchase.dt.hour
        X["purchase_day_of_week"] = purchase.dt.dayofweek
        X["purchase_is_weekend"] = (
            purchase.dt.dayofweek >= 5
        ).astype("float")

        # Remove raw timestamps after extracting useful information.
        X = X.drop(
            columns=["signup_time", "purchase_time"],
            errors="ignore",
        )

        # Remove identifiers and raw IP address.
        # ip_country, derived using the supplied IP mapping, is retained.
        X = X.drop(
            columns=["user_id", "device_id", "ip_address"],
            errors="ignore",
        )

        categorical_columns = [
            "source",
            "browser",
            "sex",
            "ip_country",
        ]

        for column in categorical_columns:
            if column in X.columns:
                X[column] = (
                    X[column]
                    .fillna("Unknown")
                    .astype(str)
                )

        return X

    raise ValueError(
        f"Unsupported dataset_id: {dataset_id!r}. "
        "Expected credit_card, paysim, or ecommerce."
    )
