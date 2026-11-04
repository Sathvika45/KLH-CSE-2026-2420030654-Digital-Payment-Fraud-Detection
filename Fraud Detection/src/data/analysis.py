"""
Analysis and reporting module for dataset validation and class-imbalance profiling.
Generates structured Markdown reports, CSV metric summaries, and distribution plots.
Ensures strict transparency: never fabricates results and explicitly tags test fixtures.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

from src.data.loader import LoadResult, load_dataset, check_raw_data_status
from src.data.validator import (
    DatasetConfig,
    ImbalanceMetrics,
    SUPPORTED_DATASETS,
    ValidationReport,
    validate_dataset,
)
from src.visualization.plots import plot_class_distribution


def generate_markdown_report(
    report: ValidationReport,
    source_paths: list,
    is_sampled: bool = False,
    sample_size: Optional[int] = None,
    sampling_method: Optional[str] = None,
    is_test_fixture: bool = False,
) -> str:
    """Produces a detailed academic Markdown validation report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics = report.imbalance_metrics

    tag = " (TEST FIXTURE - NOT RESEARCH RESULTS)" if is_test_fixture else ""

    md_lines = [
        f"# Dataset Validation & Class-Imbalance Report: {report.dataset_name}{tag}",
        "",
        f"**Generated:** {timestamp}  ",
        f"**Target Column:** `{report.target_column}`  ",
        f"**Integrity Status:** {'PASSED (Valid)' if report.is_valid else 'FAILED (Validation Errors Detected)'}  ",
        f"**Data Scope:** {'Sampled (' + str(sample_size) + ' rows via ' + str(sampling_method) + ')' if is_sampled else 'Full Dataset (Zero Truncation)'}  ",
        "",
        "## 1. Source Files",
        "",
    ]

    for p in source_paths:
        p_obj = Path(p)
        size_mb = (p_obj.stat().st_size / (1024 * 1024)) if p_obj.exists() else 0.0
        md_lines.append(f"- `{p_obj.name}` ({size_mb:.2f} MB) - `{p_obj.resolve()}`")

    md_lines.extend([
        "",
        "## 2. Structural & Integrity Diagnostics",
        "",
        "| Diagnostic Metric | Value |",
        "| :--- | :--- |",
        f"| Total Records Evaluated | {report.total_rows:,} |",
        f"| Total Attributes (Columns) | {report.total_columns:,} |",
        f"| Duplicate Rows Detected | {report.duplicate_rows_count:,} ({(report.duplicate_rows_count / max(1, report.total_rows)) * 100:.2f}%) |",
        f"| Missing Target Values | {report.missing_target_count:,} |",
        f"| Total Missing Feature Values | {report.total_missing_values:,} |",
        f"| Columns with Missing Values | {len(report.columns_with_missing):,} |",
        "",
    ])

    if metrics:
        md_lines.extend([
            "## 3. Class Imbalance Quantification",
            "",
            "| Class Name | Label | Transaction Count | Percentage |",
            "| :--- | :--- | :--- | :--- |",
            f"| Legitimate (Majority) | `0` | {metrics.legitimate_count:,} | {metrics.legitimate_percentage:.4f}% |",
            f"| Fraudulent (Minority) | `1` | {metrics.fraud_count:,} | {metrics.fraud_percentage:.4f}% |",
            f"| **Total** | - | **{metrics.total_samples:,}** | **100.00%** |",
            "",
            f"- **Imbalance Ratio:** {metrics.imbalance_ratio:,.2f} : 1 (legitimate transactions per fraud transaction)",
            f"- **Class Balance Assessment:** {'Balanced' if metrics.is_balanced else 'Severe Class Imbalance Detected'}",
            "",
            "> [!IMPORTANT]",
            "> Resampling (such as SMOTE or controlled undersampling) must **only** be fitted on training folds.",
            "> Validation and test partitions must retain their natural class distribution to avoid data leakage.",
            "",
        ])

    if report.warnings:
        md_lines.extend([
            "## 4. Warnings & Observations",
            "",
        ])
        for w in report.warnings:
            md_lines.append(f"- [WARNING] {w}")
        md_lines.append("")

    if report.errors:
        md_lines.extend([
            "## 5. Critical Validation Errors",
            "",
        ])
        for e in report.errors:
            md_lines.append(f"- [ERROR] {e}")
        md_lines.append("")

    return "\n".join(md_lines)


def run_dataset_analysis(
    dataset_id: str,
    data_dir: Optional[Path] = None,
    custom_filepath: Optional[Path] = None,
    target_column_override: Optional[str] = None,
    sample_size: Optional[int] = None,
    random_state: int = 42,
    is_test_fixture: bool = False,
) -> Dict[str, Any]:
    """
    Executes end-to-end dataset validation, metrics calculation, report saving, and plotting.

    Returns:
        Dictionary containing validation report, metrics, report path, and figure path.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    reports_dir = project_root / "outputs" / "reports"
    figures_dir = project_root / "outputs" / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    load_result = load_dataset(
        dataset_id=dataset_id,
        data_dir=data_dir,
        custom_filepath=custom_filepath,
        sample_size=sample_size,
        random_state=random_state,
    )

    df = load_result.df
    config = load_result.dataset_config
    target_col = target_column_override or config.target_column

    # 2. Validate dataset
    validation_report = validate_dataset(
        df=df,
        target_column=target_col,
        dataset_name=config.display_name,
        expected_classes=config.expected_classes,
    )

    # 3. Save metrics summary CSV
    prefix = f"{dataset_id}_test_fixture" if is_test_fixture else dataset_id
    metrics_csv_path = reports_dir / f"{prefix}_imbalance_metrics.csv"
    if validation_report.imbalance_metrics:
        metrics_dict = validation_report.imbalance_metrics.to_dict()
        metrics_dict["dataset_id"] = dataset_id
        metrics_dict["display_name"] = config.display_name
        metrics_dict["target_column"] = target_col
        metrics_dict["total_rows_evaluated"] = validation_report.total_rows
        metrics_dict["duplicate_rows"] = validation_report.duplicate_rows_count
        metrics_dict["total_missing_values"] = validation_report.total_missing_values
        metrics_dict["is_sampled"] = load_result.is_sampled
        metrics_dict["is_test_fixture"] = is_test_fixture
        pd.DataFrame([metrics_dict]).to_csv(metrics_csv_path, index=False)

    # 4. Generate & save Markdown report
    sample_info_str = None
    if load_result.is_sampled:
        sample_info_str = f"Sampled {load_result.sample_size:,} records via {load_result.sampling_method}"

    md_report_content = generate_markdown_report(
        report=validation_report,
        source_paths=load_result.source_paths,
        is_sampled=load_result.is_sampled,
        sample_size=load_result.sample_size,
        sampling_method=load_result.sampling_method,
        is_test_fixture=is_test_fixture,
    )
    report_md_path = reports_dir / f"{prefix}_imbalance_report.md"
    report_md_path.write_text(md_report_content, encoding="utf-8")

    # 5. Generate & save publication figure
    figure_path = None
    if validation_report.imbalance_metrics:
        figure_dest = figures_dir / f"{prefix}_class_distribution.png"
        figure_path = plot_class_distribution(
            metrics=validation_report.imbalance_metrics,
            dataset_name=config.display_name + (" [Test Fixture]" if is_test_fixture else ""),
            output_path=figure_dest,
            sample_info=sample_info_str,
        )

    return {
        "validation_report": validation_report,
        "load_result": load_result,
        "metrics_csv_path": metrics_csv_path,
        "report_md_path": report_md_path,
        "figure_path": figure_path,
    }

