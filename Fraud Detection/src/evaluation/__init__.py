"""
Comprehensive multi-metric evaluation, validation threshold tuning, and model explainability:
- Metrics: Precision, Recall, F1, PR-AUC, ROC-AUC, MCC, False Positive Rate (FPR), Confusion Matrices
- Threshold Tuning: Validation-only decision threshold selection
- Explainability: SHAP (TreeSHAP & Linear SHAP)
- Efficiency & Scalability: Training time, inference latency per 1,000 transactions, memory footprint
"""

from src.evaluation.metrics import (
    compute_evaluation_metrics,
    find_optimal_threshold,
    evaluate_model_pipeline,
)

__all__ = [
    "compute_evaluation_metrics",
    "find_optimal_threshold",
    "evaluate_model_pipeline",
]
