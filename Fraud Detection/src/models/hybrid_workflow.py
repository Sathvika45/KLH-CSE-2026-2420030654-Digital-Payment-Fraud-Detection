
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

# PaySim uses a 100,000-row sample to keep the experiment practical.
DEFAULT_SAMPLE_SIZES = {
    "credit_card": None,
    "paysim": 100000,
    "ecommerce": None,
}


def calculate_metrics(y_true, probabilities, threshold=0.5):
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
        "mcc": matthews_corrcoef(
            y_true, predictions
        ),
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


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

    return float(thresholds[int(np.argmax(f1_values))])


def prepare_hybrid_data(df, dataset_id, target_column):
    """
    Separates the target before feature engineering.
    This prevents the target from becoming an input feature.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found "
            f"for dataset '{dataset_id}'."
        )

    y = df[target_column].reset_index(drop=True)
    X = df.drop(columns=[target_column]).copy()

    X = engineer_features(X, dataset_id)

    if target_column in X.columns:
        raise RuntimeError(
            "Target leakage detected: target column remains in features."
        )

    prepared_df = X.reset_index(drop=True).copy()
    prepared_df[target_column] = y

    prepared_df, prep_summary = prepare_data(
        prepared_df,
        target_column=target_column,
        drop_duplicates=True,
    )

    return prepared_df, prep_summary


def run_hybrid_workflow(
    dataset_id="ecommerce",
    sample_size=None,
    random_state=42,
):
    """
    Runs the RF + LightGBM + Logistic Regression stacking hybrid.

    Dataset IDs:
        credit_card
        paysim
        ecommerce
    """

    if dataset_id not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset: {dataset_id}. "
            f"Choose from: {list(SUPPORTED_DATASETS.keys())}"
        )

    if sample_size is None:
        sample_size = DEFAULT_SAMPLE_SIZES[dataset_id]

    config = SUPPORTED_DATASETS[dataset_id]
    target_column = config.target_column

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    suffix = dataset_id

    csv_path = REPORT_DIR / f"hybrid_{suffix}_results.csv"
    report_path = REPORT_DIR / f"hybrid_{suffix}_report.md"
    model_path = MODEL_DIR / f"hybrid_{suffix}.joblib"

    # Never overwrite the existing Credit Card hybrid artifacts.
    if dataset_id == "credit_card" and (
        csv_path.exists() or report_path.exists() or model_path.exists()
    ):
        raise FileExistsError(
            "Existing Credit Card hybrid artifacts were found. "
            "They are protected from overwrite. "
            "Use a separate backup and explicit overwrite procedure "
            "if you intentionally want to replace them."
        )

    print(f"\nLoading dataset: {config.display_name}")
    print(f"Requested sample size: {sample_size or 'Full dataset'}")

    load_result = load_dataset(
        dataset_id=dataset_id,
        sample_size=sample_size,
        random_state=random_state,
    )

    df = load_result.df.copy()

    print(f"Loaded rows: {len(df):,}")
    print(f"Sampled: {load_result.is_sampled}")

    print("\nPreparing features and target...")
    prepared_df, prep_summary = prepare_hybrid_data(
        df,
        dataset_id,
        target_column,
    )

    print(f"Rows after preparation: {len(prepared_df):,}")
    print(f"Duplicates removed: {prep_summary.dropped_duplicates:,}")

    print("\nCreating stratified 60/20/20 splits...")
    splits = create_stratified_splits(
        prepared_df,
        target_column=target_column,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        random_state=random_state,
    )

    valid, errors = verify_split_integrity(splits)

    if not valid:
        raise RuntimeError(
            f"Split integrity verification failed: {errors}"
        )

    X_train, y_train = splits.X_train, splits.y_train
    X_val, y_val = splits.X_val, splits.y_val
    X_test, y_test = splits.X_test, splits.y_test

    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")
    print(f"Test rows: {len(X_test):,}")

    print("\nBuilding Random Forest pipeline...")
    rf_model = get_baseline_model(
        "random_forest",
        random_state=random_state,
    )
    rf_pipeline = build_baseline_pipeline(X_train, rf_model)

    print("Building LightGBM pipeline...")
    lgbm_model = get_baseline_model(
        "lightgbm",
        random_state=random_state,
    )
    lgbm_pipeline = build_baseline_pipeline(X_train, lgbm_model)

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

    print("\nTraining proposed stacking hybrid...")
    start_time = time.perf_counter()
    hybrid.fit(X_train, y_train)
    training_time = time.perf_counter() - start_time

    print(f"Training completed in {training_time:.2f} seconds")

    print("\nSelecting threshold using validation data...")
    validation_probabilities = hybrid.predict_proba(X_val)[:, 1]

    best_threshold = find_best_threshold(
        y_val,
        validation_probabilities,
    )

    print(f"Validation-selected threshold: {best_threshold:.6f}")

    print("\nEvaluating on untouched test data...")
    inference_start = time.perf_counter()

    test_probabilities = hybrid.predict_proba(X_test)[:, 1]

    inference_time = time.perf_counter() - inference_start

    inference_ms_per_1000 = (
        inference_time / max(1, len(X_test))
    ) * 1000 * 1000

    default_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        threshold=0.5,
    )

    tuned_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        threshold=best_threshold,
    )

    results = pd.DataFrame([
        {
            "dataset": dataset_id,
            "dataset_name": config.display_name,
            "evaluation": "default_threshold",
            **default_metrics,
        },
        {
            "dataset": dataset_id,
            "dataset_name": config.display_name,
            "evaluation": "validation_tuned_threshold",
            **tuned_metrics,
        },
    ])

    results["training_time_sec"] = training_time
    results["inference_ms_per_1000"] = inference_ms_per_1000
    results["sample_size"] = len(df)
    results["prepared_rows"] = len(prepared_df)
    results["train_rows"] = len(X_train)
    results["validation_rows"] = len(X_val)
    results["test_rows"] = len(X_test)

    results.to_csv(csv_path, index=False)

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("# Proposed Hybrid Model Evaluation\n\n")
        file.write(
            "Architecture: Random Forest + LightGBM base learners "
            "with Logistic Regression as the stacking meta-classifier.\n\n"
        )
        file.write("## Experimental setup\n\n")
        file.write(f"- Dataset: {config.display_name}\n")
        file.write(f"- Loaded rows: {len(df):,}\n")
        file.write(f"- Prepared rows: {len(prepared_df):,}\n")
        file.write(f"- Target column: {target_column}\n")
        file.write("- Split: Stratified 60/20/20\n")
        file.write(f"- Random state: {random_state}\n")
        file.write("- Stacking: 3-fold cross-validation\n")
        file.write("- Threshold selection: Validation set only\n")
        file.write("- Final evaluation: Untouched test set\n\n")
        file.write("## Test results\n\n")
        file.write(results.to_markdown(index=False))
        file.write("\n\n## Methodological note\n\n")
        file.write(
            "The test set was not used for threshold selection. "
            "The tuned threshold was selected using validation "
            "predictions and then frozen for test evaluation.\n"
        )

    joblib.dump(hybrid, model_path)

    print("\nHybrid experiment completed!")
    print(f"Results CSV: {csv_path}")
    print(f"Report: {report_path}")
    print(f"Saved model: {model_path}")

    print("\nTest metrics:")
    print(results.to_string(index=False))

    return {
        "results": results,
        "csv_path": csv_path,
        "report_path": report_path,
        "model_path": model_path,
        "training_time_sec": training_time,
    }


if __name__ == "__main__":
    run_hybrid_workflow(dataset_id="ecommerce")
