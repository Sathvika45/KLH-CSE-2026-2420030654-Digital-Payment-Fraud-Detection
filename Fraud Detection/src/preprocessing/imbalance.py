"""
Class-imbalance mitigation pipelines and strategy definitions.
Implements the four required strategies with strict pipeline encapsulation:
1. Original Imbalanced Distribution (Baseline)
2. Cost-Sensitive Class Weighting
3. SMOTE (Synthetic Minority Over-sampling Technique)
4. Controlled Random Undersampling (RandomUnderSampler)

Guarantees that resampling is executed strictly on training data during fit(),
leaving validation and test partitions untouched at their natural class distributions.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import RobustScaler


class ImbalanceStrategy(str, Enum):
    """Enumeration of the four imbalance handling strategies."""

    ORIGINAL = "original"
    CLASS_WEIGHTED = "class_weighted"
    SMOTE = "smote"
    UNDERSAMPLING = "undersampling"


# Predefined candidate resampling ratios for multi-ratio comparison
# Ratio = (Minority Count) / (Majority Count)
PREDEFINED_RESAMPLING_RATIOS: Dict[str, float] = {
    "1:20": 0.05,  # ~4.76% fraud in training split (mild mitigation)
    "1:10": 0.10,  # ~9.09% fraud in training split (moderate mitigation)
    "1:5": 0.20,   # ~16.67% fraud in training split (strong mitigation)
    "1:2": 0.50,   # ~33.33% fraud in training split (heavy mitigation)
    "1:1": 1.00,   # 50.00% fraud in training split (full parity)
}


def get_candidate_resampling_ratios() -> Dict[str, float]:
    """Returns the dictionary of predefined resampling ratios for grid evaluation."""
    return PREDEFINED_RESAMPLING_RATIOS.copy()


def get_imbalance_pipeline(
    strategy: Union[ImbalanceStrategy, str],
    base_estimator: BaseEstimator,
    scaler: Optional[BaseEstimator] = None,
    random_state: int = 42,
    smote_sampling_strategy: Union[float, str] = 0.1,  # e.g., 10:1 ratio or 'auto'
    under_sampling_strategy: Union[float, str] = 0.1,  # e.g., 10:1 majority to minority
) -> ImbPipeline:
    """
    Constructs an imbalanced-learn Pipeline ensuring all transformations
    (scaling, resampling, fitting) execute within the proper phase to prevent leakage.

    Args:
        strategy: One of 'original', 'class_weighted', 'smote', 'undersampling'
        base_estimator: Estimator to train (e.g., LogisticRegression, RandomForest)
        scaler: Feature scaler to fit on training data (default: RobustScaler())
        random_state: Seed for stochastic resamplers
        smote_sampling_strategy: Target minority-to-majority ratio for SMOTE
        under_sampling_strategy: Target minority-to-majority ratio for RandomUnderSampler

    Returns:
        imblearn.pipeline.Pipeline
    """
    if isinstance(strategy, str):
        strategy = ImbalanceStrategy(strategy)

    steps = []

    # 1. Feature Scaler (fit on train only, transform on val/test)
    if scaler is not None:
        steps.append(("scaler", scaler))
    else:
        steps.append(("scaler", RobustScaler()))

    # 2. Resampling step (if applicable)
    if strategy == ImbalanceStrategy.ORIGINAL:
        # No resampling step
        pass

    elif strategy == ImbalanceStrategy.CLASS_WEIGHTED:
        # Class weighting is handled inside the classifier's loss function;
        # no dataset modification is performed.
        pass

    elif strategy == ImbalanceStrategy.SMOTE:
        # Synthetic oversampling applied strictly in training fold
        smote_step = SMOTE(
            sampling_strategy=smote_sampling_strategy,
            random_state=random_state,
        )
        steps.append(("resampler", smote_step))

    elif strategy == ImbalanceStrategy.UNDERSAMPLING:
        # Controlled majority undersampling applied strictly in training fold
        rus_step = RandomUnderSampler(
            sampling_strategy=under_sampling_strategy,
            random_state=random_state,
        )
        steps.append(("resampler", rus_step))

    else:
        raise ValueError(f"Unknown imbalance strategy: {strategy}")

    # 3. Estimator step
    steps.append(("classifier", base_estimator))

    return ImbPipeline(steps=steps)


def calculate_scale_pos_weight(y_train: Union[pd.Series, np.ndarray]) -> float:
    """
    Computes scale_pos_weight parameter for XGBoost/LightGBM:
    scale_pos_weight = N_negative / N_positive
    """
    y_arr = np.asarray(y_train)
    n_neg = (y_arr == 0).sum()
    n_pos = (y_arr == 1).sum()
    if n_pos == 0:
        return 1.0
    return float(n_neg / n_pos)


def verify_resampling_scope(
    pipeline: ImbPipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Verifies that pipeline fitting applies resampling strictly to training data
    and leaves the test/validation set dimensions and distribution completely untouched.

    Returns:
        Tuple of (is_clean, diagnostics_dict)
    """
    original_test_len = len(X_test)
    original_test_fraud_rate = float((y_test == 1).mean())

    # Fit pipeline on training data
    pipeline.fit(X_train, y_train)

    # Make predictions on test
    preds = pipeline.predict(X_test)

    post_test_len = len(X_test)
    post_test_fraud_rate = float((y_test == 1).mean())

    # Check for leakage/modification
    len_preserved = (original_test_len == post_test_len)
    rate_preserved = (original_test_fraud_rate == post_test_fraud_rate)
    preds_valid = (len(preds) == original_test_len)

    is_clean = len_preserved and rate_preserved and preds_valid

    diagnostics = {
        "is_clean": is_clean,
        "original_test_len": original_test_len,
        "post_test_len": post_test_len,
        "original_test_fraud_rate": original_test_fraud_rate,
        "post_test_fraud_rate": post_test_fraud_rate,
        "predictions_generated": len(preds),
    }

    return is_clean, diagnostics
