"""
Baseline classification models for fraud detection benchmark.
Implements the five required baseline architectures:
1. Logistic Regression (L2-regularized linear baseline)
2. Decision Tree (CART non-linear weak learner)
3. Random Forest (Bagging ensemble)
4. XGBoost (Gradient boosting with second-order Taylor optimization)
5. LightGBM (Gradient boosting with GOSS & EFB)

Configured for the Original Imbalanced Baseline (zero resampling, zero class weighting).
"""

from typing import Any, Dict, List, Optional
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Optional gradient-boosting libraries with graceful fallbacks
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    xgb = None
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    lgb = None
    HAS_LIGHTGBM = False

BASELINE_MODEL_REGISTRY = {
    "logistic_regression": {
        "display_name": "Logistic Regression",
        "description": "L2-regularized linear classifier with lbfgs solver",
        "class": LogisticRegression,
        "is_available": True,
    },
    "decision_tree": {
        "display_name": "Decision Tree (CART)",
        "description": "Standard recursive binary splitting decision tree",
        "class": DecisionTreeClassifier,
        "is_available": True,
    },
    "random_forest": {
        "display_name": "Random Forest",
        "description": "Bagged ensemble of de-correlated decision trees",
        "class": RandomForestClassifier,
        "is_available": True,
    },
    "xgboost": {
        "display_name": "XGBoost",
        "description": "Extreme gradient boosting with histogram tree method",
        "class": xgb.XGBClassifier if HAS_XGBOOST else None,
        "is_available": HAS_XGBOOST,
    },
    "lightgbm": {
        "display_name": "LightGBM",
        "description": "Light gradient boosting with leaf-wise tree growth",
        "class": lgb.LGBMClassifier if HAS_LIGHTGBM else None,
        "is_available": HAS_LIGHTGBM,
    },
}


def get_baseline_model(
    model_name: str,
    random_state: int = 42,
    **kwargs: Any,
) -> BaseEstimator:
    """
    Instantiates a baseline model configured for the original imbalanced benchmark
    (unweighted loss function, fixed random seed).

    Args:
        model_name: One of 'logistic_regression', 'decision_tree', 'random_forest', 'xgboost', 'lightgbm'
        random_state: Reproducible seed
        kwargs: Hyperparameter overrides

    Returns:
        Instantiated scikit-learn compatible estimator
    """
    model_name = model_name.lower().strip()
    if model_name not in BASELINE_MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported baseline model '{model_name}'. "
            f"Supported models: {list(BASELINE_MODEL_REGISTRY.keys())}"
        )

    if model_name == "logistic_regression":
        params = {
            "max_iter": 1000,
            "random_state": random_state,
            "solver": "lbfgs",
            "class_weight": None,  # Original imbalanced baseline
        }
        params.update(kwargs)
        return LogisticRegression(**params)

    elif model_name == "decision_tree":
        params = {
            "random_state": random_state,
            "max_depth": 10,
            "min_samples_split": 5,
            "class_weight": None,  # Original imbalanced baseline
        }
        params.update(kwargs)
        return DecisionTreeClassifier(**params)

    elif model_name == "random_forest":
        params = {
            "n_estimators": 100,
            "max_depth": 12,
            "random_state": random_state,
            "n_jobs": -1,
            "class_weight": None,  # Original imbalanced baseline
        }
        params.update(kwargs)
        return RandomForestClassifier(**params)

    elif model_name == "xgboost":
        if not HAS_XGBOOST or xgb is None:
            raise ImportError(
                "XGBoost is not installed or unavailable in this Python environment. "
                "Install it via 'pip install xgboost' to enable the XGBoost baseline."
            )
        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": random_state,
            "eval_metric": "logloss",
            "scale_pos_weight": 1.0,  # Unweighted baseline
            "n_jobs": -1,
        }
        params.update(kwargs)
        return xgb.XGBClassifier(**params)

    elif model_name == "lightgbm":
        if not HAS_LIGHTGBM or lgb is None:
            raise ImportError(
                "LightGBM is not installed or unavailable in this Python environment. "
                "Install it via 'pip install lightgbm' to enable the LightGBM baseline."
            )
        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": random_state,
            "is_unbalance": False,  # Unweighted baseline
            "scale_pos_weight": 1.0,
            "verbose": -1,
            "n_jobs": -1,
        }
        params.update(kwargs)
        return lgb.LGBMClassifier(**params)

    raise RuntimeError(f"Unhandled model configuration for '{model_name}'.")


def get_all_baseline_models(
    random_state: int = 42,
    only_available: bool = False,
    **kwargs: Any,
) -> Dict[str, BaseEstimator]:
    """
    Returns a dictionary of all 5 instantiated baseline estimators for the unweighted benchmark.
    If only_available is True, omits models whose optional dependencies are missing.
    """
    models = {}
    for name, meta in BASELINE_MODEL_REGISTRY.items():
        if only_available and not meta["is_available"]:
            continue
        models[name] = get_baseline_model(name, random_state=random_state, **kwargs)
    return models

