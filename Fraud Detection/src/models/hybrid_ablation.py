
from pathlib import Path
import time

import joblib
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
    precision_recall_curve,
)

from src.preprocessing.features import engineer_features
from src.preprocessing.split import (
    prepare_data,
    create_stratified_splits,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
MODEL_PATH = ROOT / "outputs" / "models" / "hybrid_credit_card.joblib"
REPORT_DIR = ROOT / "outputs" / "reports"


def find_best_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    if len(thresholds) == 0:
        return 0.5

    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    return float(thresholds[np.argmax(f1_values)])


def evaluate_model(name, y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "model": name,
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
        "average_precision": average_precision_score(
            y_true, probabilities
        ),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "mcc": matthews_corrcoef(y_true, predictions),
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def run_ablation():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    df = engineer_features(df, "credit_card")

    prepared_df, _ = prepare_data(
        df,
        target_column="Class",
        drop_duplicates=True,
    )

    print("Recreating original data split...")
    splits = create_stratified_splits(
        prepared_df,
        target_column="Class",
        random_state=42,
    )

    print("Loading saved stacking model...")
    hybrid = joblib.load(MODEL_PATH)

    components = hybrid.named_estimators_

    models = {
        "Random Forest": components["random_forest"],
        "LightGBM": components["lightgbm"],
        "Stacking Hybrid": hybrid,
    }

    results = []

    for name, model in models.items():
        print(f"\nEvaluating {name}...")

        validation_probabilities = model.predict_proba(
            splits.X_val
        )[:, 1]

        threshold = find_best_threshold(
            splits.y_val,
            validation_probabilities,
        )

        start = time.perf_counter()

        test_probabilities = model.predict_proba(
            splits.X_test
        )[:, 1]

        elapsed = time.perf_counter() - start

        inference_ms_per_1000 = (
            elapsed / len(splits.X_test)
        ) * 1_000_000

        row = evaluate_model(
            name,
            splits.y_test,
            test_probabilities,
            threshold,
        )

        row["inference_ms_per_1000"] = inference_ms_per_1000
        results.append(row)

        print(
            f"Validation threshold: {threshold:.6f} | "
            f"Test F1: {row['f1']:.4f} | "
            f"AP: {row['average_precision']:.4f}"
        )

    results_df = pd.DataFrame(results)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORT_DIR / "hybrid_ablation_results.csv"
    results_df.to_csv(output_path, index=False)

    print("\nABLATION RESULTS")
    print(results_df.to_string(index=False))
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    run_ablation()
