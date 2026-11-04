
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
    thresholds = np.linspace(0.01, 0.99, 200)

    scores = [
        f1_score(
            y_true,
            (probabilities >= threshold).astype(int),
            zero_division=0,
        )
        for threshold in thresholds
    ]

    return float(thresholds[int(np.argmax(scores))])


def run_weighted_lightgbm():
    print("Loading PaySim sample...")

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
    weight = negatives / positives

    print(f"Training rows: {len(X_train)}")
    print(f"Non-fraud cases: {negatives}")
    print(f"Fraud cases: {positives}")
    print(f"Calculated scale_pos_weight: {weight:.4f}")

    # Original unweighted LightGBM
    unweighted_model = get_baseline_model("lightgbm")
    unweighted_pipeline = build_baseline_pipeline(
        X_train, unweighted_model
    )

    # Weighted LightGBM
    weighted_model = get_baseline_model("lightgbm")
    weighted_model.set_params(scale_pos_weight=weight)

    weighted_pipeline = build_baseline_pipeline(
        X_train, weighted_model
    )

    results = []

    for name, pipeline in [
        ("LightGBM_Unweighted", unweighted_pipeline),
        ("LightGBM_Weighted", weighted_pipeline),
    ]:
        print(f"\nTraining {name}...")

        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        training_time = time.perf_counter() - start

        val_probabilities = pipeline.predict_proba(X_val)[:, 1]
        threshold = find_best_threshold(
            y_val, val_probabilities
        )

        test_probabilities = pipeline.predict_proba(X_test)[:, 1]

        default_metrics = evaluate(
            y_test, test_probabilities, 0.5
        )
        tuned_metrics = evaluate(
            y_test, test_probabilities, threshold
        )

        row = {
            "model": name,
            "training_time_sec": training_time,
            "validation_threshold": threshold,
        }

        for key, value in default_metrics.items():
            row[f"default_{key}"] = value

        for key, value in tuned_metrics.items():
            row[f"tuned_{key}"] = value

        results.append(row)

        print(f"Training time: {training_time:.2f} seconds")
        print(f"Validation threshold: {threshold:.4f}")
        print(f"Test PR-AUC: {tuned_metrics['pr_auc']:.6f}")
        print(f"Test ROC-AUC: {tuned_metrics['roc_auc']:.6f}")
        print(f"Test F1: {tuned_metrics['f1']:.6f}")
        print(f"Test recall: {tuned_metrics['recall']:.6f}")
        print(f"Test precision: {tuned_metrics['precision']:.6f}")

    output_dir = Path("outputs/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    output_path = output_dir / "paysim_weighted_lightgbm.csv"
    results_df.to_csv(output_path, index=False)

    print("\nExperiment completed.")
    print(f"Results saved to: {output_path}")
    print("\nComparison:")
    print(
        results_df[
            [
                "model",
                "default_pr_auc",
                "tuned_pr_auc",
                "tuned_precision",
                "tuned_recall",
                "tuned_f1",
                "tuned_mcc",
                "tuned_fpr",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    run_weighted_lightgbm()
