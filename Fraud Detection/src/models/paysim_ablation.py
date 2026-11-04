
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
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
from src.data.validator import SUPPORTED_DATASETS
from src.preprocessing.features import engineer_features
from src.preprocessing.split import (
    prepare_data,
    create_stratified_splits,
    verify_split_integrity,
)
from src.models.baselines import get_baseline_model
from src.models.workflow import build_baseline_pipeline


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "outputs" / "reports"
MODEL_DIR = ROOT / "outputs" / "models"


def prepare_data_for_ablation(df, dataset_id, target_column):
    if target_column not in df.columns:
        raise ValueError(f"Target column {target_column} not found.")

    y = df[target_column].reset_index(drop=True)
    X = df.drop(columns=[target_column]).copy()

    X = engineer_features(X, dataset_id)

    if target_column in X.columns:
        raise RuntimeError("Target leakage detected.")

    prepared = X.reset_index(drop=True).copy()
    prepared[target_column] = y

    prepared, summary = prepare_data(
        prepared,
        target_column=target_column,
        drop_duplicates=True,
    )

    return prepared, summary


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


def evaluate_predictions(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
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
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def run_paysim_ablation(random_state=42):
    dataset_id = "paysim"
    config = SUPPORTED_DATASETS[dataset_id]
    target_column = config.target_column

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    output_csv = REPORT_DIR / "paysim_ablation_results.csv"
    output_report = REPORT_DIR / "paysim_ablation_report.md"
    output_models = MODEL_DIR / "paysim_ablation_models.joblib"

    if output_csv.exists() or output_report.exists() or output_models.exists():
        raise FileExistsError(
            "Ablation output files already exist. "
            "Back them up or rename them before rerunning."
        )

    print("\nLoading PaySim sample...")
    load_result = load_dataset(
        dataset_id=dataset_id,
        sample_size=100000,
        random_state=random_state,
    )

    df = load_result.df.copy()
    print(f"Loaded rows: {len(df):,}")

    print("\nPreparing data...")
    prepared, prep_summary = prepare_data_for_ablation(
        df, dataset_id, target_column
    )

    print(f"Prepared rows: {len(prepared):,}")
    print(f"Duplicates removed: {prep_summary.dropped_duplicates:,}")

    print("\nCreating stratified splits...")
    splits = create_stratified_splits(
        prepared,
        target_column=target_column,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        random_state=random_state,
    )

    valid, errors = verify_split_integrity(splits)

    if not valid:
        raise RuntimeError(f"Split integrity failed: {errors}")

    X_train, y_train = splits.X_train, splits.y_train
    X_val, y_val = splits.X_val, splits.y_val
    X_test, y_test = splits.X_test, splits.y_test

    print(
        f"Train: {len(X_train):,} | "
        f"Validation: {len(X_val):,} | "
        f"Test: {len(X_test):,}"
    )

    print("\nBuilding individual models...")

    rf_pipeline = build_baseline_pipeline(
        X_train,
        get_baseline_model("random_forest", random_state),
    )

    lgbm_pipeline = build_baseline_pipeline(
        X_train,
        get_baseline_model("lightgbm", random_state),
    )

    meta_classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
    )

    hybrid = StackingClassifier(
        estimators=[
            ("random_forest", rf_pipeline),
            ("lightgbm", lgbm_pipeline),
        ],
        final_estimator=meta_classifier,
        stack_method="predict_proba",
        cv=3,
        n_jobs=1,
        passthrough=False,
    )

    experiments = [
        ("Random Forest", rf_pipeline),
        ("LightGBM", lgbm_pipeline),
        ("RF + LightGBM Stacking", hybrid),
    ]

    results = []
    fitted_models = {}

    for model_name, model in experiments:
        print(f"\nTraining: {model_name}")

        start = time.perf_counter()
        model.fit(X_train, y_train)
        training_time = time.perf_counter() - start

        print(f"Training time: {training_time:.2f} seconds")

        print("Selecting threshold on validation data...")
        val_probabilities = model.predict_proba(X_val)[:, 1]

        threshold = find_best_threshold(
            y_val, val_probabilities
        )

        print(f"Selected threshold: {threshold:.6f}")

        print("Evaluating on test data...")
        start = time.perf_counter()
        test_probabilities = model.predict_proba(X_test)[:, 1]
        inference_time = time.perf_counter() - start

        inference_ms_per_1000 = (
            inference_time / len(X_test) * 1_000_000
        )

        default_metrics = evaluate_predictions(
            y_test, test_probabilities, 0.5
        )

        tuned_metrics = evaluate_predictions(
            y_test, test_probabilities, threshold
        )

        results.append({
            "model": model_name,
            "training_time_sec": training_time,
            "inference_ms_per_1000": inference_ms_per_1000,
            "validation_threshold": threshold,
            "test_default_f1": default_metrics["f1"],
            "test_tuned_f1": tuned_metrics["f1"],
            "test_default_pr_auc": default_metrics["pr_auc"],
            "test_tuned_pr_auc": tuned_metrics["pr_auc"],
            "test_default_roc_auc": default_metrics["roc_auc"],
            "test_tuned_roc_auc": tuned_metrics["roc_auc"],
            "test_tuned_precision": tuned_metrics["precision"],
            "test_tuned_recall": tuned_metrics["recall"],
            "test_tuned_mcc": tuned_metrics["mcc"],
            "test_tuned_fpr": tuned_metrics["fpr"],
            "test_tuned_tp": tuned_metrics["tp"],
            "test_tuned_fp": tuned_metrics["fp"],
            "test_tuned_tn": tuned_metrics["tn"],
            "test_tuned_fn": tuned_metrics["fn"],
        })

        fitted_models[model_name] = model

        print(
            f"PR-AUC: {tuned_metrics['pr_auc']:.5f} | "
            f"F1: {tuned_metrics['f1']:.5f} | "
            f"MCC: {tuned_metrics['mcc']:.5f}"
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

    with open(output_report, "w", encoding="utf-8") as file:
        file.write("# PaySim Hybrid Ablation Study\n\n")
        file.write("## Experimental setup\n\n")
        file.write("- Dataset: PaySim\n")
        file.write("- Sample size: 100,000 rows\n")
        file.write("- Split: Stratified 60/20/20\n")
        file.write(f"- Random state: {random_state}\n")
        file.write("- Threshold selection: Validation set only\n")
        file.write("- Final evaluation: Untouched test set\n\n")
        file.write("## Compared models\n\n")
        file.write("1. Random Forest\n")
        file.write("2. LightGBM\n")
        file.write("3. Random Forest + LightGBM stacking\n\n")
        file.write("## Results\n\n")
        file.write(results_df.to_markdown(index=False))
        file.write(
            "\n\n## Interpretation note\n\n"
            "The ablation study compares individual base learners "
            "against the stacking model using the same data split. "
            "The results indicate whether the combination provides "
            "an improvement over its individual components.\n"
        )

    joblib.dump(fitted_models, output_models)

    print("\nAblation experiment completed!")
    print(f"Results: {output_csv}")
    print(f"Report: {output_report}")
    print(f"Models: {output_models}")

    print("\nFINAL COMPARISON")
    print(results_df.to_string(index=False))

    return results_df


if __name__ == "__main__":
    run_paysim_ablation()
