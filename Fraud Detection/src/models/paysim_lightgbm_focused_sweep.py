
from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)

from src.data.loader import load_dataset
from src.preprocessing.features import engineer_features
from src.preprocessing.split import (
    prepare_data,
    create_stratified_splits,
)
from src.models.baselines import get_baseline_model
from src.models.workflow import build_baseline_pipeline


MIN_PRECISION = 0.40
WEIGHTS = [5, 10, 15, 20]


def evaluate(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": threshold,
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "pr_auc": average_precision_score(
            y_true, probabilities
        ),
        "roc_auc": roc_auc_score(
            y_true, probabilities
        ),
        "mcc": matthews_corrcoef(y_true, predictions),
        "fpr": fp / (fp + tn) if (fp + tn) else 0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def choose_threshold(y_true, probabilities):
    # Include every unique predicted probability as a candidate.
    # Predict positive when probability >= threshold.
    candidates = np.unique(probabilities)

    valid_candidates = []

    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_true, predictions, zero_division=0
        )
        recall = recall_score(
            y_true, predictions, zero_division=0
        )

        if precision >= MIN_PRECISION and predictions.sum() > 0:
            valid_candidates.append(
                (float(recall), float(precision), float(threshold))
            )

    if not valid_candidates:
        return None

    # Maximize recall, then precision.
    # If tied, prefer the higher threshold.
    valid_candidates.sort(
        key=lambda item: (item[0], item[1], item[2]),
        reverse=True,
    )

    return valid_candidates[0][2]


def run_focused_sweep():
    print("Loading PaySim dataset...")

    data = load_dataset(
        "paysim",
        sample_size=100000,
        random_state=42,
    ).df

    y = data["isFraud"].reset_index(drop=True)

    X = engineer_features(
        data.drop(columns=["isFraud"]).copy(),
        "paysim",
    )

    prepared = X.copy()
    prepared["isFraud"] = y

    prepared, _ = prepare_data(
        prepared,
        target_column="isFraud",
        drop_duplicates=True,
    )

    splits = create_stratified_splits(
        prepared,
        target_column="isFraud",
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        random_state=42,
    )

    X_train = splits.X_train
    X_val = splits.X_val
    X_test = splits.X_test

    y_train = splits.y_train
    y_val = splits.y_val
    y_test = splits.y_test

    results = []

    for weight in WEIGHTS:
        print(f"\nTraining LightGBM with weight {weight}...")

        model = get_baseline_model("lightgbm")
        model.set_params(scale_pos_weight=weight)

        pipeline = build_baseline_pipeline(X_train, model)

        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        training_time = time.perf_counter() - start

        val_probabilities = pipeline.predict_proba(X_val)[:, 1]

        threshold = choose_threshold(
            y_val, val_probabilities
        )

        if threshold is None:
            print(
                "No validation threshold achieved "
                f"precision >= {MIN_PRECISION:.0%}."
            )
            continue

        val_metrics = evaluate(
            y_val, val_probabilities, threshold
        )

        test_probabilities = pipeline.predict_proba(X_test)[:, 1]

        test_metrics = evaluate(
            y_test, test_probabilities, threshold
        )

        row = {
            "scale_pos_weight": weight,
            "training_time_sec": training_time,
            "validation_threshold": threshold,
            "validation_precision": val_metrics["precision"],
            "validation_recall": val_metrics["recall"],
            "validation_f1": val_metrics["f1"],
        }

        for key, value in test_metrics.items():
            row[f"test_{key}"] = value

        results.append(row)

        print(
            f"Validation precision: {val_metrics['precision']:.4f} | "
            f"Validation recall: {val_metrics['recall']:.4f}"
        )
        print(
            f"Test PR-AUC: {test_metrics['pr_auc']:.6f} | "
            f"Test precision: {test_metrics['precision']:.4f} | "
            f"Test recall: {test_metrics['recall']:.4f} | "
            f"Test F1: {test_metrics['f1']:.4f}"
        )

    output_dir = Path("outputs/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)

    output_path = (
        output_dir / "paysim_lightgbm_focused_sweep.csv"
    )
    results_df.to_csv(output_path, index=False)

    print("\nFocused experiment completed.")
    print(f"Results saved to: {output_path}")

    if not results_df.empty:
        columns = [
            "scale_pos_weight",
            "validation_threshold",
            "validation_precision",
            "validation_recall",
            "test_pr_auc",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_mcc",
            "test_fpr",
        ]

        print("\nFinal comparison:")
        print(results_df[columns].to_string(index=False))


if __name__ == "__main__":
    run_focused_sweep()
