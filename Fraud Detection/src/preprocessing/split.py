"""
Data preparation, deduplication audit, and stratified partitioning.
Implements reproducible data preparation preserving raw datasets,
with strict verification of zero partition overlap and natural test distribution.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DuplicateAuditReport:
    """Detailed diagnostic report on duplicate rows in a dataset."""

    total_rows: int
    total_columns: int
    exact_duplicate_rows: int
    duplicate_percentage: float
    duplicate_by_class: Dict[Any, int]
    rows_involved_in_duplication: int  # count including original occurrences
    implication_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "exact_duplicate_rows": self.exact_duplicate_rows,
            "duplicate_percentage": round(self.duplicate_percentage, 4),
            "duplicate_by_class": self.duplicate_by_class,
            "rows_involved_in_duplication": self.rows_involved_in_duplication,
            "implication_summary": self.implication_summary,
        }


@dataclass
class DataPreparationSummary:
    """Audit of data preparation steps and row counts."""

    raw_row_count: int
    raw_class_counts: Dict[Any, int]
    cleaned_row_count: int
    cleaned_class_counts: Dict[Any, int]
    dropped_duplicates: int
    dropped_fraud_duplicates: int
    dropped_legitimate_duplicates: int
    deduplication_applied: bool
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_row_count": self.raw_row_count,
            "raw_class_counts": self.raw_class_counts,
            "cleaned_row_count": self.cleaned_row_count,
            "cleaned_class_counts": self.cleaned_class_counts,
            "dropped_duplicates": self.dropped_duplicates,
            "dropped_fraud_duplicates": self.dropped_fraud_duplicates,
            "dropped_legitimate_duplicates": self.dropped_legitimate_duplicates,
            "deduplication_applied": self.deduplication_applied,
            "notes": self.notes,
        }


@dataclass
class SplitArtifacts:
    """Container for partitioned data splits and partitioning metadata."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    random_state: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_samples(self) -> int:
        return len(self.X_train) + len(self.X_val) + len(self.X_test)


def audit_duplicates(
    df: pd.DataFrame, target_column: str = "Class"
) -> DuplicateAuditReport:
    """
    Analyzes exact duplicate records across ALL columns and stratifies them by target class.
    
    Notes on causal ambiguity:
        Exact duplication across all 31 dimensions indicates identical feature vectors,
        but statistical examination of the raw data cannot establish WHY the records were repeated
        (e.g., transactional network retries, payment gateway re-submissions, batch reconciliation anomalies,
        or automated card-testing probes).
    """
    total_rows = len(df)
    total_columns = len(df.columns)

    # Exact full-row duplicates across all columns (excluding first occurrence)
    dup_mask = df.duplicated(subset=None, keep="first")
    exact_duplicate_rows = int(dup_mask.sum())
    dup_pct = (exact_duplicate_rows / max(1, total_rows)) * 100.0

    # All rows involved in duplication (including the original instances)
    all_dup_mask = df.duplicated(subset=None, keep=False)
    rows_involved = int(all_dup_mask.sum())

    # Breakdown by class
    duplicate_by_class: Dict[Any, int] = {}
    if target_column in df.columns and exact_duplicate_rows > 0:
        dup_rows = df[dup_mask]
        class_counts = dup_rows[target_column].value_counts().to_dict()
        duplicate_by_class = {k: int(v) for k, v in class_counts.items()}

    implication = (
        f"Found {exact_duplicate_rows} duplicate rows ({dup_pct:.2f}% of total). "
        "Exact duplication does not establish why the records were repeated (e.g. transactional "
        "network retry, payment gateway re-submission, batch reconciliation anomaly, or fraudster probe). "
        "Retaining duplicates in machine learning risks train-test contamination and optimistic performance inflation. "
        "Removing duplicates only in the prepared data eliminates identical vector repetition while leaving "
        "data/raw/creditcard.csv untouched."
    )

    return DuplicateAuditReport(
        total_rows=total_rows,
        total_columns=total_columns,
        exact_duplicate_rows=exact_duplicate_rows,
        duplicate_percentage=dup_pct,
        duplicate_by_class=duplicate_by_class,
        rows_involved_in_duplication=rows_involved,
        implication_summary=implication,
    )


def prepare_data(
    df: pd.DataFrame,
    target_column: str = "Class",
    drop_duplicates: bool = True,
) -> Tuple[pd.DataFrame, DataPreparationSummary]:
    """
    Applies documented, reproducible data preparation without modifying or overwriting raw data.

    Args:
        df: Raw DataFrame (not modified in place)
        target_column: Name of target label
        drop_duplicates: Whether to drop exact full-row duplicates

    Returns:
        Tuple of (prepared_df, preparation_summary)
    """
    raw_row_count = len(df)
    raw_class_counts = (
        df[target_column].value_counts().to_dict() if target_column in df.columns else {}
    )

    if drop_duplicates:
        # Check all columns for exact duplication
        dup_mask = df.duplicated(subset=None, keep="first")
        dropped_duplicates = int(dup_mask.sum())

        dup_rows = df[dup_mask]
        dropped_fraud = (
            int((dup_rows[target_column] == 1).sum()) if target_column in df.columns else 0
        )
        dropped_legit = (
            int((dup_rows[target_column] == 0).sum()) if target_column in df.columns else 0
        )

        # Drop duplicates and reset index
        prepared_df = df[~dup_mask].copy().reset_index(drop=True)
        note = (
            f"Deduplicated {dropped_duplicates} exact full-row records across all {len(df.columns)} columns. "
            f"Raw dataset is untouched. Note: exact duplication does not establish the operational cause of repetition."
        )
    else:
        # Preserve all records (for sensitivity benchmarking)
        prepared_df = df.copy().reset_index(drop=True)
        dropped_duplicates = 0
        dropped_fraud = 0
        dropped_legit = 0
        note = "Deduplication disabled; all rows preserved for duplicate-retention sensitivity analysis."

    cleaned_row_count = len(prepared_df)

    # Invariant assertion: cleaned count must strictly equal raw minus dropped
    assert cleaned_row_count == raw_row_count - dropped_duplicates, (
        f"Row count inconsistency: raw={raw_row_count}, dropped={dropped_duplicates}, cleaned={cleaned_row_count}"
    )

    cleaned_class_counts = (
        prepared_df[target_column].value_counts().to_dict()
        if target_column in prepared_df.columns
        else {}
    )

    summary = DataPreparationSummary(
        raw_row_count=raw_row_count,
        raw_class_counts={k: int(v) for k, v in raw_class_counts.items()},
        cleaned_row_count=cleaned_row_count,
        cleaned_class_counts={k: int(v) for k, v in cleaned_class_counts.items()},
        dropped_duplicates=dropped_duplicates,
        dropped_fraud_duplicates=dropped_fraud,
        dropped_legitimate_duplicates=dropped_legit,
        deduplication_applied=drop_duplicates,
        notes=note,
    )

    return prepared_df, summary


def create_stratified_splits(
    df: pd.DataFrame,
    target_column: str = "Class",
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    random_state: int = 42,
) -> SplitArtifacts:
    """
    Splits DataFrame into Train (60%), Validation (20%), and Test (20%) sets
    using stratified sampling to preserve the exact natural class ratio in all partitions.

    Raises:
        ValueError: If split ratios do not sum to 1.0 or target_column is missing.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio:.4f}"
        )

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' missing from DataFrame.")

    # Reset index to guarantee monotonic indexing from 0 to N-1
    df_indexed = df.reset_index(drop=True)

    X = df_indexed.drop(columns=[target_column]).copy()
    y = df_indexed[target_column].copy()

    # Total remaining ratio for val + test
    eval_ratio = val_ratio + test_ratio  # e.g., 0.40

    # Stage 1: Split into Train and Temp (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=eval_ratio,
        stratify=y,
        random_state=random_state,
        shuffle=True,
    )

    # Stage 2: Split Temp into Validation and Test
    # Fraction of Temp that goes to Test: test_ratio / eval_ratio (e.g. 0.20 / 0.40 = 0.50)
    relative_test_ratio = test_ratio / eval_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_ratio,
        stratify=y_temp,
        random_state=random_state,
        shuffle=True,
    )

    # Invariant assertion: split counts must sum to original row count
    total_split_rows = len(X_train) + len(X_val) + len(X_test)
    assert total_split_rows == len(df_indexed), (
        f"Partition size conservation error: {total_split_rows} != {len(df_indexed)}"
    )

    # Track original index positions for overlap auditing
    train_indices = np.array(X_train.index)
    val_indices = np.array(X_val.index)
    test_indices = np.array(X_test.index)

    metadata = {
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "train_fraud_count": int((y_train == 1).sum()),
        "val_fraud_count": int((y_val == 1).sum()),
        "test_fraud_count": int((y_test == 1).sum()),
        "train_fraud_rate": float((y_train == 1).mean()),
        "val_fraud_rate": float((y_val == 1).mean()),
        "test_fraud_rate": float((y_test == 1).mean()),
        "random_state": random_state,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }

    return SplitArtifacts(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        random_state=random_state,
        metadata=metadata,
    )


def verify_split_integrity(splits: SplitArtifacts) -> Tuple[bool, List[str]]:
    """
    Performs rigorous checks on the split artifacts:
    1. Zero index overlap between train, val, and test.
    2. Sum of partition sizes equals total size.
    3. Natural class distribution preservation in validation and test sets.
    """
    errors: List[str] = []

    train_set = set(splits.train_indices)
    val_set = set(splits.val_indices)
    test_set = set(splits.test_indices)

    # 1. Overlap checks
    train_val_overlap = train_set.intersection(val_set)
    if train_val_overlap:
        errors.append(f"Data leakage detected: {len(train_val_overlap)} overlapping indices between Train and Val.")

    train_test_overlap = train_set.intersection(test_set)
    if train_test_overlap:
        errors.append(f"Data leakage detected: {len(train_test_overlap)} overlapping indices between Train and Test.")

    val_test_overlap = val_set.intersection(test_set)
    if val_test_overlap:
        errors.append(f"Data leakage detected: {len(val_test_overlap)} overlapping indices between Val and Test.")

    # 2. Total count conservation
    total_partitioned = len(train_set) + len(val_set) + len(test_set)
    if total_partitioned != splits.total_samples:
        errors.append(
            f"Index count mismatch: partitioned={total_partitioned} vs total_samples={splits.total_samples}."
        )

    # 3. Class distribution check (stratification accuracy)
    train_rate = splits.metadata.get("train_fraud_rate", 0.0)
    val_rate = splits.metadata.get("val_fraud_rate", 0.0)
    test_rate = splits.metadata.get("test_fraud_rate", 0.0)

    # Check relative difference (within 10% relative variance)
    if not np.isclose(val_rate, train_rate, rtol=0.10):
        errors.append(f"Stratification discrepancy in Val: train_rate={train_rate:.5f}, val_rate={val_rate:.5f}")

    if not np.isclose(test_rate, train_rate, rtol=0.10):
        errors.append(f"Stratification discrepancy in Test: train_rate={train_rate:.5f}, test_rate={test_rate:.5f}")

    is_valid = len(errors) == 0
    return is_valid, errors
