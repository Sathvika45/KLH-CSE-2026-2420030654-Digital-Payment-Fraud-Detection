"""
Data loading, schema alignment, validation, and integrity verification modules.
Supports Credit Card Fraud, IEEE-CIS Fraud Detection, and PaySim datasets.
"""

from src.data.validator import (
    DatasetConfig,
    ImbalanceMetrics,
    ValidationReport,
    SUPPORTED_DATASETS,
    validate_target_column,
    calculate_imbalance_metrics,
    validate_dataset,
)
from src.data.loader import (
    LoadResult,
    load_dataset,
    check_raw_data_status,
    get_default_data_dir,
)
from src.data.analysis import (
    run_dataset_analysis,
    generate_markdown_report,
)

__all__ = [
    "DatasetConfig",
    "ImbalanceMetrics",
    "ValidationReport",
    "SUPPORTED_DATASETS",
    "validate_target_column",
    "calculate_imbalance_metrics",
    "validate_dataset",
    "LoadResult",
    "load_dataset",
    "check_raw_data_status",
    "get_default_data_dir",
    "run_dataset_analysis",
    "generate_markdown_report",
]
