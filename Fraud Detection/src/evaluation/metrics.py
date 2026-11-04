"""
Evaluation metrics computation, validation-only decision threshold optimization,
and leakage-free evaluation protocols.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_evaluation_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_proba: Union[pd.Series, np.ndarray],
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Computes the full multi-dimensional fraud evaluation metrics:
    Precision, Recall, F1, PR-AUC, ROC-AUC, MCC, FPR, and Confusion Matrix.

    Args:
        y_true: Ground truth binary labels (0 or 1)
        y_proba: Predicted probability of the positive/fraud class (y=1)
        threshold: Decision threshold for converting probabilities to binary predictions

    Returns:
        Dictionary of computed metrics
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_proba, dtype=float)

    # Generate binary predictions based on decision threshold
    y_pred = (y_p >= threshold).astype(int)

    # Confusion matrix elements: tn, fp, fn, tp
    cm = confusion_matrix(y_t, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Core rates
    precision = float(precision_score(y_t, y_pred, zero_division=0))
    recall = float(recall_score(y_t, y_pred, zero_division=0))
    f1 = float(f1_score(y_t, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_t, y_pred))

    # False Positive Rate: FP / (FP + TN)
    fpr = float(fp / max(1, fp + tn))

    # Curve AUCs (computed across continuous probabilities)
    try:
        pr_auc = float(average_precision_score(y_t, y_p))
    except Exception:
        pr_auc = 0.0

    try:
        roc_auc = float(roc_auc_score(y_t, y_p))
    except Exception:
        roc_auc = 0.5

    return {
        "threshold": round(threshold, 4),
        "total_samples": int(len(y_t)),
        "precision": round(precision, 5),
        "recall": round(recall, 5),
        "f1": round(f1, 5),
        "pr_auc": round(pr_auc, 5),
        "roc_auc": round(roc_auc, 5),
        "mcc": round(mcc, 5),
        "false_positive_rate": round(fpr, 6),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def find_optimal_threshold(
    y_val: Union[pd.Series, np.ndarray],
    y_val_proba: Union[pd.Series, np.ndarray],
    metric: str = "f1",
    num_thresholds: int = 99,
) -> Tuple[float, float, Dict[float, float]]:
    """
    Selects the decision threshold using validation data only.

    Uses thresholds from the precision-recall curve instead of
    restricting the search to an arbitrary fixed interval.
    """

    from sklearn.metrics import precision_recall_curve

    y_v = np.asarray(y_val, dtype=int)
    y_p = np.asarray(y_val_proba, dtype=float)

    if len(y_v) != len(y_p):
        raise ValueError("Validation labels and probabilities must have equal length.")

    if not np.isfinite(y_p).all():
        raise ValueError("Validation probabilities contain invalid values.")

    precision, recall, thresholds = precision_recall_curve(y_v, y_p)

    if len(thresholds) == 0:
        return 0.5, 0.0, {}

    if metric == "f1":
        scores_array = (
            2 * precision[:-1] * recall[:-1]
            / (precision[:-1] + recall[:-1] + 1e-12)
        )

    elif metric == "mcc":
        scores_array = np.array([
            matthews_corrcoef(
                y_v,
                (y_p >= threshold).astype(int)
            )
            for threshold in thresholds
        ])

    else:
        raise ValueError(
            f"Unsupported threshold optimization metric: {metric}"
        )

    best_index = int(np.argmax(scores_array))

    scores = {
        float(threshold): float(score)
        for threshold, score in zip(thresholds, scores_array)
    }

    return (
        float(thresholds[best_index]),
        float(scores_array[best_index]),
        scores,
    )



def evaluate_model_pipeline(
    pipeline: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tune_threshold_metric: str = "f1",
) -> Dict[str, Any]:
    """
    Executes a strictly leakage-free training and multi-stage evaluation protocol:
    1. Fits pipeline strictly on (X_train, y_train).
    2. Generates validation probabilities and tunes decision threshold strictly on y_val.
    3. Evaluates test set using the FROZEN threshold selected from validation.
    4. Guarantees test features and labels are never seen during fitting or threshold tuning.

    Returns:
        Dictionary with validation metrics, test metrics, and the frozen optimal threshold.
    """
    # 1. Fit strictly on train
    pipeline.fit(X_train, y_train)

    # 2. Validation evaluation & threshold tuning
    if hasattr(pipeline, "predict_proba"):
        val_proba = pipeline.predict_proba(X_val)[:, 1]
    else:
        # For non-probabilistic models, use decision_function scaled to [0, 1]
        decision = pipeline.decision_function(X_val)
        val_proba = 1.0 / (1.0 + np.exp(-decision))

    opt_threshold, best_val_score, _ = find_optimal_threshold(
        y_val=y_val,
        y_val_proba=val_proba,
        metric=tune_threshold_metric,
    )
    val_metrics = compute_evaluation_metrics(
        y_true=y_val,
        y_proba=val_proba,
        threshold=opt_threshold,
    )

    # 3. Test evaluation using frozen threshold from validation
    if hasattr(pipeline, "predict_proba"):
        test_proba = pipeline.predict_proba(X_test)[:, 1]
    else:
        decision = pipeline.decision_function(X_test)
        test_proba = 1.0 / (1.0 + np.exp(-decision))

    test_metrics = compute_evaluation_metrics(
        y_true=y_test,
        y_proba=test_proba,
        threshold=opt_threshold,  # FROZEN threshold from validation
    )

    return {
        "frozen_optimal_threshold": opt_threshold,
        "tuning_metric": tune_threshold_metric,
        "val_score_at_threshold": best_val_score,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

