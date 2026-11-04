"""
Training and evaluation workflow for the Original Imbalanced Baseline.
Coordinates data preparation, stratified partitioning, leakage-free pipeline fitting,
validation-only threshold optimization, and final test set benchmarking.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from src.preprocessing.features import engineer_features
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.evaluation.metrics import (
    compute_evaluation_metrics,
    find_optimal_threshold,
)
from src.models.baselines import (
    BASELINE_MODEL_REGISTRY,
    get_baseline_model,
)

from src.preprocessing.split import (
    DataPreparationSummary,
    SplitArtifacts,
    create_stratified_splits,
    prepare_data,
    verify_split_integrity,
)


@dataclass
class ModelBenchmarkResult:
    """Benchmark results for a single classifier across default and tuned thresholds."""

    model_key: str
    display_name: str
    train_time_sec: float
    test_inference_ms_per_1k: float
    val_default_metrics: Dict[str, Any]
    val_optimal_threshold: float
    val_tuned_metrics: Dict[str, Any]
    test_default_metrics: Dict[str, Any]
    test_tuned_metrics: Dict[str, Any]

    def to_flat_dict(self) -> Dict[str, Any]:
        """Flattens metrics into a single dictionary for tabular logging."""
        vd = self.val_default_metrics
        vt = self.val_tuned_metrics
        td = self.test_default_metrics
        tt = self.test_tuned_metrics

        return {
            "model_key": self.model_key,
            "display_name": self.display_name,
            "train_time_sec": round(self.train_time_sec, 3),
            "test_ms_per_1k": round(self.test_inference_ms_per_1k, 3),
            # Default threshold (0.50) on Test Set
            "test_default_threshold": td["threshold"],
            "test_default_ap": td["pr_auc"],
            "test_default_pr_auc": td["pr_auc"],
            "test_default_precision": td["precision"],
            "test_default_recall": td["recall"],
            "test_default_f1": td["f1"],
            "test_default_roc_auc": td["roc_auc"],
            "test_default_mcc": td["mcc"],
            "test_default_fpr": td["false_positive_rate"],
            "test_default_tp": td["confusion_matrix"]["tp"],
            "test_default_fp": td["confusion_matrix"]["fp"],
            "test_default_fn": td["confusion_matrix"]["fn"],
            "test_default_tn": td["confusion_matrix"]["tn"],
            # Validation-Tuned Threshold on Test Set
            "frozen_val_threshold": tt["threshold"],
            "test_tuned_ap": tt["pr_auc"],
            "test_tuned_pr_auc": tt["pr_auc"],
            "test_tuned_precision": tt["precision"],
            "test_tuned_recall": tt["recall"],
            "test_tuned_f1": tt["f1"],
            "test_tuned_roc_auc": tt["roc_auc"],
            "test_tuned_mcc": tt["mcc"],
            "test_tuned_fpr": tt["false_positive_rate"],
            "test_tuned_tp": tt["confusion_matrix"]["tp"],
            "test_tuned_fp": tt["confusion_matrix"]["fp"],
            "test_tuned_fn": tt["confusion_matrix"]["fn"],
            "test_tuned_tn": tt["confusion_matrix"]["tn"],
            # Validation Reference Scores
            "val_default_f1": vd["f1"],
            "val_tuned_f1": vt["f1"],
            "val_default_mcc": vd["mcc"],
            "val_tuned_mcc": vt["mcc"],
        }


def generate_baseline_markdown_report(
    dataset_name: str,
    prep_summary: DataPreparationSummary,
    splits: SplitArtifacts,
    results: List[ModelBenchmarkResult],
    is_test_fixture: bool = False,
) -> str:
    """Generates an academic Markdown report for the Original Imbalanced Baseline experiment."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tag = " [TEST FIXTURE - NOT RESEARCH RESULTS]" if is_test_fixture else ""

    md = [
        f"# Benchmark Report: Original Imbalanced Baseline{tag}",
        f"**Dataset:** {dataset_name}  ",
        f"**Strategy:** Original Class Distribution (No Resampling, No Class Weighting)  ",
        f"**Generated:** {timestamp}  ",
        "",
        "## 1. Experimental Design & Leakage-Free Protocol",
        "",
        "- **Data Integrity:** Raw CSV is completely untouched. Deduplication executed in memory.",
        f"- **Raw Record Count:** {prep_summary.raw_row_count:,} ({prep_summary.raw_class_counts.get(1, 0):,} fraud, {prep_summary.raw_class_counts.get(0, 0):,} legitimate).",
        f"- **Deduplicated Record Count:** {prep_summary.cleaned_row_count:,} ({prep_summary.dropped_duplicates:,} exact full-row duplicates removed across all columns).",
        "- **Stratified 60/20/20 Partitioning (Fixed Seed 42):**",
        f"  - **Training Split:** {len(splits.X_train):,} samples ({splits.metadata['train_fraud_count']:,} fraud, {splits.metadata['train_fraud_rate']*100:.3f}%).",
        f"  - **Validation Split:** {len(splits.X_val):,} samples ({splits.metadata['val_fraud_count']:,} fraud, {splits.metadata['val_fraud_rate']*100:.3f}%).",
        f"  - **Test Split:** {len(splits.X_test):,} samples ({splits.metadata['test_fraud_count']:,} fraud, {splits.metadata['test_fraud_rate']*100:.3f}%).",
        "- **Zero Leakage Enforcement:** Numeric imputation, RobustScaler, and categorical imputation/OneHotEncoder are fitted strictly on training data. Resamplers omitted. Decision thresholds optimized strictly on validation data and applied frozen to test data.",
        "",
        "## 2. Test Set Performance: Default Decision Threshold (θ = 0.50)",
        "",
        "| Classifier | Average Precision (AP) | Recall | Precision | F1-Score | MCC | ROC-AUC | FPR | TP | FP | FN | TN |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in results:
        td = r.test_default_metrics
        cm = td["confusion_matrix"]
        md.append(
            f"| **{r.display_name}** | {td['pr_auc']:.4f} | {td['recall']:.4f} | {td['precision']:.4f} | "
            f"{td['f1']:.4f} | {td['mcc']:.4f} | {td['roc_auc']:.4f} | {td['false_positive_rate']:.6f} | "
            f"{cm['tp']} | {cm['fp']} | {cm['fn']} | {cm['tn']:,} |"
        )

    md.extend([
        "",
        "## 3. Test Set Performance: Validation-Tuned Decision Threshold (Frozen θ*val)",
        "",
        "| Classifier | Optimal θ*val | Average Precision (AP) | Recall | Precision | F1-Score | MCC | ROC-AUC | FPR | TP | FP | FN | TN |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for r in results:
        tt = r.test_tuned_metrics
        cm = tt["confusion_matrix"]
        md.append(
            f"| **{r.display_name}** | {r.val_optimal_threshold:.4f} | {tt['pr_auc']:.4f} | {tt['recall']:.4f} | {tt['precision']:.4f} | "
            f"{tt['f1']:.4f} | {tt['mcc']:.4f} | {tt['roc_auc']:.4f} | {tt['false_positive_rate']:.6f} | "
            f"{cm['tp']} | {cm['fp']} | {cm['fn']} | {cm['tn']:,} |"
        )

    md.extend([
        "",
        "## 4. Operational Latency and Computational Cost",
        "",
        "| Classifier | Training Duration (s) | Test Inference Latency (ms / 1,000 tx) |",
        "| :--- | :---: | :---: |",
    ])

    for r in results:
        md.append(
            f"| **{r.display_name}** | {r.train_time_sec:.3f} s | {r.test_inference_ms_per_1k:.2f} ms |"
        )

    md.extend([
        "",
        "## 5. Methodological Observations",
        "",
        "- **Impact of Extreme Skewness:** Without class weighting or resampling, models with uncalibrated probabilities frequently optimize for global accuracy, depressing Recall at the default threshold θ = 0.50.",
        "- **Value of Validation-Tuned Thresholds:** Tuning θ strictly on validation data allows models to navigate the precision-recall trade-off without risking test-set data leakage.",
        "- **Next Comparative Stage:** These original baseline scores establish the formal empirical benchmark against which Class Weighting, SMOTE, and Controlled Undersampling will be compared.",
        "",
    ])

    return "\n".join(md)
def build_baseline_pipeline(X_train: pd.DataFrame, estimator):
    """
    Builds a leakage-safe preprocessing and classification pipeline.

    Numeric features:
        Median imputation + RobustScaler

    Categorical features:
        Most-frequent imputation + OneHotEncoder

    All preprocessing is fitted only on training data.
    """

    numeric_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ])

        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])

        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", estimator),
    ])

def run_baseline_workflow(
    df: pd.DataFrame,
    dataset_id: str,
    dataset_name: str = "Credit Card Fraud Detection",
    target_column: str = "Class",
    models_to_run: Optional[List[str]] = None,
    drop_duplicates: bool = True,
    random_state: int = 42,
    tune_threshold_metric: str = "f1",
    output_dir: Optional[Path] = None,
    save_reports: bool = True,
    is_test_fixture: bool = False,
) -> Dict[str, Any]:

    """
    Executes the training and evaluation workflow for the Original Imbalanced Baseline:
    1. Prepares dataset (removes duplicates in memory, preserves raw dataset).
    2. Splits into stratified Train (60%), Validation (20%), and Test (20%).
    3. Builds preprocessing pipelines with numeric scaling and categorical encoding, fitted on training data only.
    4. Evaluates default threshold (0.50) vs. validation-tuned threshold on test set.
    5. Saves structured CSV and Markdown reports under outputs/reports/.

    Args:
        df: Input DataFrame (e.g. from creditcard.csv or synthetic fixture)
        dataset_name: Display name
        target_column: Name of binary target
        models_to_run: List of model keys to run (defaults to all 5 baselines)
        drop_duplicates: Whether to drop exact full-row duplicates
        random_state: Fixed random seed (42)
        tune_threshold_metric: Metric for threshold search ('f1' or 'mcc')
        output_dir: Reports destination directory (defaults to outputs/reports/)
        save_reports: Whether to write CSV and Markdown to disk
        is_test_fixture: If True, prefixes filenames with test_fixture

    Returns:
        Dictionary containing results summary DataFrame, individual ModelBenchmarkResults,
        split artifacts, and output file paths.
    """
    if models_to_run is None:
        models_to_run = list(BASELINE_MODEL_REGISTRY.keys())

    project_root = Path(__file__).resolve().parent.parent.parent
    if output_dir is None:
        output_dir = project_root / "outputs" / "reports"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = engineer_features(df, dataset_id)

    # 1. Data Preparation (preserves raw df)
    prepared_df, prep_summary = prepare_data(
        df=df, target_column=target_column, drop_duplicates=drop_duplicates
    )

    # 2. Stratified Partitioning (60 / 20 / 20)
    splits = create_stratified_splits(
        df=prepared_df,
        target_column=target_column,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        random_state=random_state,
    )
    is_valid_split, split_errors = verify_split_integrity(splits)
    if not is_valid_split:
        raise RuntimeError(f"Split integrity verification failed: {split_errors}")

    results: List[ModelBenchmarkResult] = []

    # 3. Train & Evaluate Each Baseline Model
    for model_key in models_to_run:
        meta = BASELINE_MODEL_REGISTRY[model_key]
        display_name = meta["display_name"]

        base_estimator = get_baseline_model(
            model_key,
            random_state=random_state,
        )

        pipeline = build_baseline_pipeline(
            X_train=splits.X_train,
            estimator=base_estimator,
        )

        # Fit strictly on train
        t0 = time.perf_counter()
        pipeline.fit(splits.X_train, splits.y_train)
        train_time_sec = time.perf_counter() - t0

        # Predict validation probabilities
        val_proba = pipeline.predict_proba(splits.X_val)[:, 1]

        # Validation Default Threshold (0.50)
        val_default_metrics = compute_evaluation_metrics(
            y_true=splits.y_val, y_proba=val_proba, threshold=0.50
        )

        # Validation Threshold Optimization
        opt_thresh, best_val_score, _ = find_optimal_threshold(
            y_val=splits.y_val,
            y_val_proba=val_proba,
            metric=tune_threshold_metric,
        )
        val_tuned_metrics = compute_evaluation_metrics(
            y_true=splits.y_val, y_proba=val_proba, threshold=opt_thresh
        )

        # Predict test probabilities & measure inference latency
        t_infer_start = time.perf_counter()
        test_proba = pipeline.predict_proba(splits.X_test)[:, 1]
        t_infer_total_sec = time.perf_counter() - t_infer_start
        ms_per_1k = (t_infer_total_sec / max(1, len(splits.X_test))) * 1000.0 * 1000.0

        # Test Default Threshold (0.50)
        test_default_metrics = compute_evaluation_metrics(
            y_true=splits.y_test, y_proba=test_proba, threshold=0.50
        )

        # Test Tuned Threshold (FROZEN validation threshold opt_thresh)
        test_tuned_metrics = compute_evaluation_metrics(
            y_true=splits.y_test, y_proba=test_proba, threshold=opt_thresh
        )

        res = ModelBenchmarkResult(
            model_key=model_key,
            display_name=display_name,
            train_time_sec=train_time_sec,
            test_inference_ms_per_1k=ms_per_1k,
            val_default_metrics=val_default_metrics,
            val_optimal_threshold=opt_thresh,
            val_tuned_metrics=val_tuned_metrics,
            test_default_metrics=test_default_metrics,
            test_tuned_metrics=test_tuned_metrics,
        )
        results.append(res)

    # 4. Tabulate Results
    summary_rows = [r.to_flat_dict() for r in results]
    summary_df = pd.DataFrame(summary_rows)

    csv_path = None
    md_path = None

    if save_reports:
        prefix = f"baseline_original_{dataset_id}_test_fixture" if is_test_fixture else f"baseline_original_{dataset_id}"
        csv_path = output_dir / f"{prefix}_summary.csv"
        summary_df.to_csv(csv_path, index=False)

        md_content = generate_baseline_markdown_report(
            dataset_name=dataset_name,
            prep_summary=prep_summary,
            splits=splits,
            results=results,
            is_test_fixture=is_test_fixture,
        )
        md_path = output_dir / f"{prefix}_report.md"
        md_path.write_text(md_content, encoding="utf-8")

    return {
        "results": results,
        "summary_df": summary_df,
        "splits": splits,
        "prep_summary": prep_summary,
        "csv_path": csv_path,
        "md_path": md_path,
    }

