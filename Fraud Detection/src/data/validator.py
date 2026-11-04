"""
Dataset validation and class-imbalance metrics analysis.
Provides strict schema validation, missing value detection, duplicate checks,
and class imbalance calculation for digital payment fraud datasets.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd


@dataclass
class DatasetConfig:
    """Configuration schema for supported fraud detection datasets."""

    dataset_id: str
    display_name: str
    target_column: str
    primary_filename: str
    secondary_filename: Optional[str] = None
    join_key: Optional[str] = None
    expected_classes: Tuple[Any, ...] = (0, 1)
    notes: str = ""


# Supported datasets registry
SUPPORTED_DATASETS: Dict[str, DatasetConfig] = {
    "credit_card": DatasetConfig(
        dataset_id="credit_card",
        display_name="Credit Card Fraud Detection",
        target_column="Class",
        primary_filename="creditcard.csv",
        expected_classes=(0, 1),
        notes="28 PCA numerical features (V1-V28) plus Time and Amount.",
    ),
    "paysim": DatasetConfig(
        dataset_id="paysim",
        display_name="PaySim Mobile Money Simulation",
        target_column="isFraud",
        primary_filename="paysim.csv",
        expected_classes=(0, 1),
        notes="Synthetic mobile money transactions with transaction types and balance features.",
    ),
    "ecommerce": DatasetConfig(
        dataset_id="ecommerce",
        display_name="Fraud E-commerce",
        target_column="class",
        primary_filename="Fraud_Data.csv",
        secondary_filename="IpAddress_to_Country.csv",
        expected_classes=(0, 1),
        notes="E-commerce transactions with user, device, browser, purchase-time, and IP-country information.",
    ),
}

# Alternative common filenames for auto-discovery
FILENAME_ALIASES: Dict[str, str] = {
    "creditcard.csv": "credit_card",
    "credit_card.csv": "credit_card",
    "creditcard_fraud.csv": "credit_card",
    "paysim.csv": "paysim",
    "ps_20174392719_1491204439457_log.csv": "paysim",
    "fraud_data.csv": "ecommerce",
}


@dataclass
class ImbalanceMetrics:
    """Quantitative class-imbalance summary metrics."""

    total_samples: int
    legitimate_count: int
    fraud_count: int
    legitimate_percentage: float
    fraud_percentage: float
    imbalance_ratio: float  # Non-fraud to fraud ratio (e.g. 577.88 means ~578:1)
    target_classes: List[Any]
    is_balanced: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "legitimate_count": self.legitimate_count,
            "fraud_count": self.fraud_count,
            "legitimate_percentage": round(self.legitimate_percentage, 4),
            "fraud_percentage": round(self.fraud_percentage, 4),
            "imbalance_ratio": round(self.imbalance_ratio, 2),
            "target_classes": self.target_classes,
            "is_balanced": self.is_balanced,
        }


@dataclass
class ValidationReport:
    """Comprehensive integrity and validation diagnostics for a dataset."""

    dataset_name: str
    target_column: str
    is_valid: bool
    total_rows: int
    total_columns: int
    duplicate_rows_count: int
    missing_target_count: int
    total_missing_values: int
    columns_with_missing: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    imbalance_metrics: Optional[ImbalanceMetrics] = None

    def to_dict(self) -> Dict[str, Any]:
        base = {
            "dataset_name": self.dataset_name,
            "target_column": self.target_column,
            "is_valid": self.is_valid,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "duplicate_rows_count": self.duplicate_rows_count,
            "missing_target_count": self.missing_target_count,
            "total_missing_values": self.total_missing_values,
            "columns_with_missing_count": len(self.columns_with_missing),
            "errors": self.errors,
            "warnings": self.warnings,
        }
        if self.imbalance_metrics:
            base["imbalance_metrics"] = self.imbalance_metrics.to_dict()
        return base


def validate_target_column(
    df: pd.DataFrame,
    target_column: str,
    expected_classes: Tuple[Any, ...] = (0, 1),
) -> Tuple[bool, List[str], List[str]]:
    """
    Validates presence, completeness, and value labels of the target column.

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    errors: List[str] = []
    warnings: List[str] = []

    if target_column not in df.columns:
        errors.append(f"Target column '{target_column}' is missing from DataFrame columns.")
        return False, errors, warnings

    target_series = df[target_column]

    # Check for missing values in target
    missing_count = int(target_series.isna().sum())
    if missing_count > 0:
        errors.append(
            f"Target column '{target_column}' contains {missing_count} missing (NaN/null) values."
        )

    # Check class labels (drop NaNs for label checking)
    valid_values = target_series.dropna()
    unique_classes = set(valid_values.unique())

    # Check for invalid / unexpected classes
    expected_set = set(expected_classes)
    unexpected = unique_classes - expected_set
    if unexpected:
        errors.append(
            f"Target column '{target_column}' contains invalid labels: {unexpected}. "
            f"Expected binary labels: {expected_classes}."
        )

    # Check if only one class exists
    if len(unique_classes) < 2:
        warnings.append(
            f"Target column '{target_column}' contains only a single class: {unique_classes}. "
            "Both legitimate (0) and fraud (1) samples are required for classification."
        )

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def calculate_imbalance_metrics(
    df: pd.DataFrame,
    target_column: str,
    legitimate_label: Any = 0,
    fraud_label: Any = 1,
) -> ImbalanceMetrics:
    """
    Computes class counts, percentages, and imbalance ratio.

    Raises:
        ValueError if target_column is missing or empty.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    target_series = df[target_column].dropna()
    total_samples = len(target_series)
    if total_samples == 0:
        raise ValueError(f"Target column '{target_column}' contains no valid rows.")

    counts = target_series.value_counts().to_dict()
    legitimate_count = int(counts.get(legitimate_label, 0))
    fraud_count = int(counts.get(fraud_label, 0))

    legitimate_pct = (legitimate_count / total_samples) * 100.0
    fraud_pct = (fraud_count / total_samples) * 100.0

    if fraud_count > 0:
        imbalance_ratio = legitimate_count / fraud_count
    else:
        imbalance_ratio = float("inf")

    # A dataset is roughly balanced if minority class is between 40% and 60%
    is_balanced = 40.0 <= fraud_pct <= 60.0

    return ImbalanceMetrics(
        total_samples=total_samples,
        legitimate_count=legitimate_count,
        fraud_count=fraud_count,
        legitimate_percentage=legitimate_pct,
        fraud_percentage=fraud_pct,
        imbalance_ratio=imbalance_ratio,
        target_classes=sorted(list(counts.keys())),
        is_balanced=is_balanced,
    )


def validate_dataset(
    df: pd.DataFrame,
    target_column: str,
    dataset_name: str = "Dataset",
    expected_classes: Tuple[Any, ...] = (0, 1),
    check_duplicates: bool = True,
) -> ValidationReport:
    """
    Executes a comprehensive validation on the given DataFrame:
    - Target column presence and validity
    - Missing value audit
    - Duplicate rows detection
    - Class-imbalance quantification
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    errors: List[str] = []
    warnings: List[str] = []

    # Target column validation
    target_valid, target_errors, target_warnings = validate_target_column(
        df=df, target_column=target_column, expected_classes=expected_classes
    )
    errors.extend(target_errors)
    warnings.extend(target_warnings)

    missing_target_count = (
        int(df[target_column].isna().sum()) if target_column in df.columns else 0
    )

    # Missing values across all columns
    missing_series = df.isna().sum()
    total_missing = int(missing_series.sum())
    cols_with_missing = {
        col: int(cnt) for col, cnt in missing_series.items() if cnt > 0
    }
    if total_missing > 0:
        warnings.append(
            f"Dataset contains {total_missing} total missing values across {len(cols_with_missing)} columns."
        )

    # Duplicate rows check
    duplicate_count = 0
    if check_duplicates:
        duplicate_count = int(df.duplicated().sum())
        if duplicate_count > 0:
            warnings.append(
                f"Dataset contains {duplicate_count} duplicate rows ({(duplicate_count / total_rows) * 100:.2f}%)."
            )

    # Imbalance metrics
    imbalance_metrics = None
    if target_valid:
        try:
            imbalance_metrics = calculate_imbalance_metrics(
                df=df, target_column=target_column
            )
        except Exception as e:
            errors.append(f"Failed to calculate imbalance metrics: {str(e)}")

    is_valid = len(errors) == 0

    return ValidationReport(
        dataset_name=dataset_name,
        target_column=target_column,
        is_valid=is_valid,
        total_rows=total_rows,
        total_columns=total_cols,
        duplicate_rows_count=duplicate_count,
        missing_target_count=missing_target_count,
        total_missing_values=total_missing,
        columns_with_missing=cols_with_missing,
        errors=errors,
        warnings=warnings,
        imbalance_metrics=imbalance_metrics,
    )

