
from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
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


def find_best_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    precision = precision[:-1]
    recall = recall[:-1]

    denominator = precision + recall

    f1_scores = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )

    best_index = int(np.argmax(f1_scores))

    return float(thresholds[best_index])


def run_weight_sweep():

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

    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())

    if positives == 0:
        raise ValueError("Training data contains no fraud cases.")

    class_ratio = negatives / positives

    weights = [1, 10, 25, 50, 100, 200, class_ratio]

    print(f"Training rows: {len(X_train)}")
    print(f"Fraud cases: {positives}")
    print(f"Non-fraud cases: {negatives}")
    print(f"Class ratio: {class_ratio:.4f}")

    results = []

    for weight in weights:

        label = (
            "Unweighted"
            if weight == 1
            else f"Weight_{weight:.2f}"
        )

        print(f"\nTraining LightGBM with weight {weight:.2f}...")

        model = get_baseline_model("lightgbm")
        model.set_params(scale_pos_weight=weight)

        pipeline = build_baseline_pipeline(X_train, model)

        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        training_time = time.perf_counter() - start

        # Validation probabilities
        val_probabilities = pipeline.predict_proba(X_val)[:, 1]

        print("\nValidation probability distribution:")
        print("Minimum:", val_probabilities.min())
        print("Maximum:", val_probabilities.max())
        print("Mean:", val_probabilities.mean())

        print(
            "Percentiles:",
            np.percentile(
                val_probabilities,
                [50, 90, 95, 99, 99.5, 99.9, 100],
            ),
        )

        print("Exactly 1.0:", np.sum(val_probabilities == 1.0))
        print("Above 0.99:", np.sum(val_probabilities >= 0.99))

        # Threshold is selected using validation data only
        threshold = find_best_threshold(
            y_val, val_probabilities
        )

        # Test probabilities
        test_probabilities = pipeline.predict_proba(X_test)[:, 1]

        default_metrics = evaluate(
            y_test, test_probabilities, 0.5
        )

        tuned_metrics = evaluate(
            y_test, test_probabilities, threshold
        )

        row = {
            "model": label,
            "scale_pos_weight": weight,
            "training_time_sec": training_time,
            "validation_threshold": threshold,
            "validation_f1": f1_score(
                y_val,
                (val_probabilities >= threshold).astype(int),
                zero_division=0,
            ),
        }

        for key, value in default_metrics.items():
            row[f"default_{key}"] = value

        for key, value in tuned_metrics.items():
            row[f"tuned_{key}"] = value

        results.append(row)

        print(
            f"Threshold: {threshold:.8f} | "
            f"PR-AUC: {tuned_metrics['pr_auc']:.6f} | "
            f"Precision: {tuned_metrics['precision']:.4f} | "
            f"Recall: {tuned_metrics['recall']:.4f} | "
            f"F1: {tuned_metrics['f1']:.4f}"
        )

    # Save results
    output_dir = Path("outputs/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)

    output_path = output_dir / "paysim_lightgbm_weight_sweep.csv"

    results_df.to_csv(output_path, index=False)

    print("\nWeight experiment completed.")
    print(f"Results saved to: {output_path}")

    columns = [
        "model",
        "scale_pos_weight",
        "validation_threshold",
        "tuned_pr_auc",
        "tuned_roc_auc",
        "tuned_precision",
        "tuned_recall",
        "tuned_f1",
        "tuned_mcc",
        "tuned_fpr",
    ]

    print("\nFinal comparison:")
    print(results_df[columns].to_string(index=False))


if __name__ == "__main__":
    run_weight_sweep()
