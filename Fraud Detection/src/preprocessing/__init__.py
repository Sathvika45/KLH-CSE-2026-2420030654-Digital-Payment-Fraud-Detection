"""
Preprocessing pipelines, scaling, encoding, and class-imbalance mitigation strategies:
- Original Imbalanced Data
- Cost-sensitive Class Weighting
- Synthetic Minority Over-sampling Technique (SMOTE)
- Controlled Undersampling (RandomUnderSampler)
Strictly engineered to prevent data leakage across train/validation/test splits.
"""

from src.preprocessing.split import (
    DuplicateAuditReport,
    DataPreparationSummary,
    SplitArtifacts,
    audit_duplicates,
    prepare_data,
    create_stratified_splits,
    verify_split_integrity,
)
from src.preprocessing.imbalance import (
    ImbalanceStrategy,
    get_imbalance_pipeline,
    calculate_scale_pos_weight,
    verify_resampling_scope,
)

__all__ = [
    "DuplicateAuditReport",
    "DataPreparationSummary",
    "SplitArtifacts",
    "audit_duplicates",
    "prepare_data",
    "create_stratified_splits",
    "verify_split_integrity",
    "ImbalanceStrategy",
    "get_imbalance_pipeline",
    "calculate_scale_pos_weight",
    "verify_resampling_scope",
]
