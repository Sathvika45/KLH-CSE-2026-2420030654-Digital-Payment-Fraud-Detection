
"""
Comparative Analysis of Fraud Detection Techniques in Digital Payment Systems
Main Pipeline Orchestrator and CLI
"""

import argparse
import importlib
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pandas as pd

from src.data.loader import (
    check_raw_data_status,
    load_dataset,
)
from src.data.validator import (
    SUPPORTED_DATASETS,
    validate_dataset,
)
from src.data.analysis import (
    run_dataset_analysis,
    generate_markdown_report,
)
from src.models.workflow import run_baseline_workflow
from src.visualization.plots import plot_class_distribution
from src.models.imbalance_experiment import main as run_imbalance_experiment
from src.models.hybrid_workflow import run_hybrid_workflow
from src.models.interpretability import run_interpretability


PROJECT_TITLE = (
    "Comparative Analysis of Fraud Detection Techniques "
    "in Digital Payment Systems"
)

REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scikit-learn", "sklearn"),
    ("imbalanced-learn", "imblearn"),
    ("lightgbm", "lightgbm"),
    ("xgboost", "xgboost"),
    ("shap", "shap"),
    ("joblib", "joblib"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("plotly", "plotly"),
    ("streamlit", "streamlit"),
]

EXPECTED_DIRS = [
    "data/raw",
    "data/processed",
    "docs",
    "notebooks",
    "outputs/figures",
    "outputs/models",
    "outputs/reports",
    "src/data",
    "src/preprocessing",
    "src/models",
    "src/evaluation",
    "src/visualization",
    "tests",
]


def print_banner():
    width = 80
    print("=" * width)
    print(PROJECT_TITLE.center(width))
    print("BTech Data Analytics and Visualisation Project".center(width))
    print("=" * width)


def check_environment() -> bool:
    """Check project directories and installed dependencies."""

    project_root = Path(__file__).resolve().parent

    print(f"\n[1] Project Root Directory: {project_root}")
    print(f"    Python Version: {sys.version.split()[0]} ({sys.executable})")

    print("\n[2] Checking Directory Structure:")

    missing_dirs = []

    for rel_path in EXPECTED_DIRS:
        directory = project_root / rel_path
        status = "OK" if directory.exists() and directory.is_dir() else "MISSING"

        print(f"    [{status}] {rel_path}")

        if status == "MISSING":
            missing_dirs.append(rel_path)

    print("\n[3] Checking Python Packages:")

    missing_packages = []

    for display_name, import_name in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", "installed")
            print(f"    [OK] {display_name:<20} version: {version}")

        except ImportError:
            print(
                f"    [NOT INSTALLED] {display_name:<20} "
                f"(import name: {import_name})"
            )
            missing_packages.append(display_name)

    print("\n" + "-" * 80)

    if missing_dirs:
        print(f"Directory Status: {len(missing_dirs)} missing directories.")
    else:
        print("Directory Status: All project directories are present.")

    if missing_packages:
        print(f"Package Status: {len(missing_packages)} package(s) missing.")
        print("Run: pip install -r requirements.txt")
    else:
        print("Package Status: All required dependencies are installed.")

    print("-" * 80 + "\n")

    return not missing_dirs and not missing_packages


def report_data_status(data_dir: Optional[Path] = None):
    """Report availability of supported datasets."""

    status = check_raw_data_status(data_dir)

    print("\n" + "=" * 80)
    print("RAW DATASET AVAILABILITY AUDIT (data/raw/)".center(80))
    print("=" * 80)

    any_found = False

    for dataset_id, info in status.items():

        state = "FOUND" if info["found"] else "NOT FOUND"

        if info["found"]:
            any_found = True

        print(f"\n- Dataset: {info['display_name']} ({dataset_id})")
        print(
            f"  Target Column:      "
            f"{SUPPORTED_DATASETS[dataset_id].target_column}"
        )
        print(f"  Status:             [{state}]")
        print(f"  Expected File:      {info['expected_primary']}")

        if info.get("expected_secondary"):
            print(f"  Secondary File:     {info['expected_secondary']}")

        if info["found"]:
            print(f"  Resolved Path:      {info['primary_path']}")

    print("\n" + "-" * 80)

    if not any_found:
        print("No raw datasets found in data/raw/.")
        print("Place the required CSV files in data/raw/.")
        print("You can also run synthetic fixtures using:")
        print("    python main.py --test-fixture all")
    else:
        print("Available datasets can be analyzed using:")
        print("    python main.py --analyze <dataset_id>")

    print("-" * 80 + "\n")

    return any_found


def generate_synthetic_fixture(
    dataset_id: str,
    n_samples: int = 1000
) -> pd.DataFrame:
    """
    Generate an in-memory synthetic fixture matching
    the required dataset schema.

    These fixtures are for pipeline testing only.
    They are not research datasets or research results.
    """

    np.random.seed(42)

    if dataset_id not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_id}")

    config = SUPPORTED_DATASETS[dataset_id]
    target = config.target_column

    if dataset_id == "credit_card":

        n_fraud = max(2, int(n_samples * 0.002))
        n_legit = n_samples - n_fraud

        labels = [0] * n_legit + [1] * n_fraud

        df = pd.DataFrame({
            "Time": np.sort(
                np.random.uniform(0, 172800, size=n_samples)
            ),
            "V1": np.random.randn(n_samples),
            "V2": np.random.randn(n_samples),
            "V3": np.random.randn(n_samples),
            "V4": np.random.randn(n_samples),
            "Amount": np.random.exponential(
                scale=88.0,
                size=n_samples
            ),
            target: labels,
        })

    elif dataset_id == "paysim":

        n_fraud = max(2, int(n_samples * 0.0013))
        n_legit = n_samples - n_fraud

        labels = [0] * n_legit + [1] * n_fraud

        types = np.random.choice(
            ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"],
            size=n_samples
        )

        df = pd.DataFrame({
            "step": np.random.randint(1, 744, size=n_samples),
            "type": types,
            "amount": np.random.exponential(
                scale=180000,
                size=n_samples
            ),
            "nameOrig": [
                f"C{i:08d}" for i in range(n_samples)
            ],
            "oldbalanceOrg": np.random.exponential(
                scale=80000,
                size=n_samples
            ),
            "newbalanceOrig": np.random.exponential(
                scale=70000,
                size=n_samples
            ),
            "nameDest": [
                f"M{i:08d}" for i in range(n_samples)
            ],
            "oldbalanceDest": np.random.exponential(
                scale=85000,
                size=n_samples
            ),
            "newbalanceDest": np.random.exponential(
                scale=95000,
                size=n_samples
            ),
            target: labels,
        })

    elif dataset_id == "ecommerce":

        n_fraud = max(5, int(n_samples * 0.09))
        n_legit = n_samples - n_fraud

        labels = [0] * n_legit + [1] * n_fraud

        signup_times = pd.date_range(
            start="2025-01-01",
            periods=n_samples,
            freq="h"
        )

        purchase_times = signup_times + pd.to_timedelta(
            np.random.randint(1, 720, size=n_samples),
            unit="h"
        )

        df = pd.DataFrame({
            "user_id": [
                f"user_{i}" for i in range(n_samples)
            ],
            "signup_time": signup_times,
            "purchase_time": purchase_times,
            "purchase_value": np.random.uniform(
                5, 500, size=n_samples
            ),
            "device_id": [
                f"device_{i % 100}" for i in range(n_samples)
            ],
            "source": np.random.choice(
                ["SEO", "Ads", "Direct"],
                size=n_samples
            ),
            "browser": np.random.choice(
                ["Chrome", "Firefox", "Safari"],
                size=n_samples
            ),
            "sex": np.random.choice(
                ["M", "F"],
                size=n_samples
            ),
            "age": np.random.randint(
                18, 70, size=n_samples
            ),
            "ip_address": np.random.uniform(
                1, 4294967295, size=n_samples
            ),
            target: labels,
        })

    else:
        raise ValueError(
            f"Unknown dataset_id '{dataset_id}' for fixture generation."
        )

    return df


def run_fixture_analysis(target_id: str):
    """Validate and visualize synthetic test fixtures."""

    targets = (
        list(SUPPORTED_DATASETS.keys())
        if target_id == "all"
        else [target_id]
    )

    project_root = Path(__file__).resolve().parent

    reports_dir = project_root / "outputs" / "reports"
    figures_dir = project_root / "outputs" / "figures"

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("RUNNING SYNTHETIC TEST FIXTURE VALIDATION".center(80))
    print(
        "NOTICE: These are test fixtures, NOT research results.".center(80)
    )
    print("=" * 80)

    for dataset_id in targets:

        config = SUPPORTED_DATASETS[dataset_id]

        print(
            f"\n--> Generating test fixture for: "
            f"{config.display_name}..."
        )

        df = generate_synthetic_fixture(
            dataset_id,
            n_samples=1000
        )

        report = validate_dataset(
            df=df,
            target_column=config.target_column,
            dataset_name=config.display_name,
            expected_classes=config.expected_classes,
        )

        metrics = report.imbalance_metrics

        print(
            f"    Validation Status: "
            f"{'PASSED' if report.is_valid else 'FAILED'}"
        )
        print(f"    Target Column:     {report.target_column}")
        print(f"    Total Samples:     {report.total_rows:,}")

        if metrics:

            print(
                f"    Legitimate:        "
                f"{metrics.legitimate_count:,} "
                f"({metrics.legitimate_percentage:.2f}%)"
            )

            print(
                f"    Fraudulent:        "
                f"{metrics.fraud_count:,} "
                f"({metrics.fraud_percentage:.2f}%)"
            )

            print(
                f"    Imbalance Ratio:   "
                f"{metrics.imbalance_ratio:.1f} : 1"
            )

        md_content = generate_markdown_report(
            report=report,
            source_paths=[f"in_memory_fixture_{dataset_id}.csv"],
            is_test_fixture=True,
        )

        md_file = (
            reports_dir
            / f"{dataset_id}_test_fixture_imbalance_report.md"
        )

        md_file.write_text(
            md_content,
            encoding="utf-8"
        )

        csv_file = (
            reports_dir
            / f"{dataset_id}_test_fixture_imbalance_metrics.csv"
        )

        if metrics:

            metrics_dict = metrics.to_dict()
            metrics_dict["dataset_id"] = dataset_id
            metrics_dict["display_name"] = config.display_name
            metrics_dict["is_test_fixture"] = True

            pd.DataFrame([metrics_dict]).to_csv(
                csv_file,
                index=False
            )

        fig_file = (
            figures_dir
            / f"{dataset_id}_test_fixture_class_distribution.png"
        )

        if metrics:

            plot_class_distribution(
                metrics=metrics,
                dataset_name=f"{config.display_name} [Test Fixture]",
                output_path=fig_file,
                sample_info="Synthetic Unit Test Fixture (1,000 records)",
            )

        print(f"    Saved Markdown Report: {md_file.relative_to(project_root)}")
        print(f"    Saved Metrics CSV:     {csv_file.relative_to(project_root)}")
        print(f"    Saved Figure:          {fig_file.relative_to(project_root)}")

    print("\n" + "-" * 80)
    print("Test fixture run completed.")
    print("-" * 80 + "\n")


def execute_analysis(
    dataset_id: str,
    sample_size: Optional[int] = None
):
    """Analyze actual dataset files in data/raw/."""

    status = check_raw_data_status()

    targets = (
        list(SUPPORTED_DATASETS.keys())
        if dataset_id == "all"
        else [dataset_id]
    )

    for d_id in targets:

        d_status = status.get(d_id, {})

        if not d_status.get("found"):

            print(f"\n[ERROR] Dataset '{d_id}' not found.")
            print(f"Expected file: {d_status.get('expected_primary')}")
            print("Place the dataset file in data/raw/.")
            continue

        print(
            f"\n--> Analyzing real dataset: "
            f"{d_status.get('display_name')}..."
        )

        try:

            result = run_dataset_analysis(
                dataset_id=d_id,
                sample_size=sample_size,
                is_test_fixture=False,
            )

            validation_report = result["validation_report"]
            metrics = validation_report.imbalance_metrics

            print(
                f"    Validation Status: "
                f"{'PASSED' if validation_report.is_valid else 'FAILED'}"
            )

            print(
                f"    Total Rows: "
                f"{validation_report.total_rows:,}"
            )

            print(
                f"    Total Columns: "
                f"{validation_report.total_columns:,}"
            )

            print(
                f"    Duplicate Rows: "
                f"{validation_report.duplicate_rows_count:,}"
            )

            if metrics:

                print(
                    f"    Legitimate Count: "
                    f"{metrics.legitimate_count:,} "
                    f"({metrics.legitimate_percentage:.4f}%)"
                )

                print(
                    f"    Fraud Count: "
                    f"{metrics.fraud_count:,} "
                    f"({metrics.fraud_percentage:.4f}%)"
                )

                print(
                    f"    Imbalance Ratio: "
                    f"{metrics.imbalance_ratio:,.2f} : 1"
                )

            print(f"    Markdown Report: {result['report_md_path']}")
            print(f"    Metrics CSV:     {result['metrics_csv_path']}")
            print(f"    Figure:          {result['figure_path']}")

        except Exception as error:

            print(
                f"[ERROR] Failed to analyze dataset "
                f"'{d_id}': {error}"
            )


def execute_baseline_experiment(
    dataset_id: str = "credit_card",
    sample_size: Optional[int] = None,
    drop_duplicates: bool = True,
):
    """Execute the original imbalanced baseline workflow."""

    status = check_raw_data_status()
    dataset_status = status.get(dataset_id, {})

    if not dataset_status.get("found"):

        print(f"\n[ERROR] Dataset '{dataset_id}' not found.")
        print(
            f"Expected file: "
            f"{dataset_status.get('expected_primary')}"
        )
        print("Ensure the dataset file is in data/raw/.")
        return

    config = SUPPORTED_DATASETS[dataset_id]

    print(
        f"\n--> Loading {config.display_name} "
        f"for Original Baseline Benchmark..."
    )

    load_result = load_dataset(
        dataset_id=dataset_id,
        sample_size=sample_size
    )

    print(
        f"    Loaded records: {load_result.loaded_records:,} "
        f"(Sampled: {load_result.is_sampled})"
    )

    print("\n--> Starting Original Imbalanced Baseline...")
    print(
        "    Models: Logistic Regression, Decision Tree, "
        "Random Forest, XGBoost, LightGBM"
    )
    print(
        "    Imbalance Strategy: Original "
        "(No Resampling, No Class Weighting)"
    )
    print(
        "    Partitions: Stratified 60% Train, "
        "20% Validation, 20% Test (Seed 42)"
    )

    workflow_result = run_baseline_workflow(
        df=load_result.df,
        dataset_id=dataset_id,
        dataset_name=config.display_name,
        target_column=config.target_column,
        drop_duplicates=drop_duplicates,
        random_state=42,
        tune_threshold_metric="f1",
        is_test_fixture=False,
    )

    print("\n" + "=" * 80)
    print("ORIGINAL IMBALANCED BASELINE BENCHMARK COMPLETE".center(80))
    print("=" * 80)

    print(f"\nSaved Metrics CSV:     {workflow_result['csv_path']}")
    print(f"Saved Markdown Report: {workflow_result['md_path']}\n")



def main():

    parser = argparse.ArgumentParser(
        description=f"{PROJECT_TITLE} - Pipeline Orchestrator"
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Check project structure and dependencies.",
    )

    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Check availability of datasets in data/raw/.",
    )

    parser.add_argument(
        "--analyze",
        type=str,
        choices=[*SUPPORTED_DATASETS.keys(), "all"],
        help="Validate and analyze dataset class imbalance.",
    )

    parser.add_argument(
        "--run-baseline",
        type=str,
        choices=["credit_card", "paysim", "ecommerce"],
        help="Run the original imbalanced baseline across five models.",
    )

    parser.add_argument(
        "--run-imbalance",
        action="store_true",
        help="Run class imbalance handling experiments.",
    )

    parser.add_argument(
        "--run-hybrid",
        action="store_true",
        help="Run the RF + LightGBM + Logistic Regression stacking experiment.",
    )

    parser.add_argument(
        "--run-interpretability",
        action="store_true",
        help="Run Random Forest SHAP explainability analysis.",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional number of rows to sample.",
    )

    parser.add_argument(
        "--test-fixture",
        type=str,
        choices=["credit_card", "paysim", "ecommerce", "all"],
        help="Run validation using synthetic test fixtures only.",
    )

    parser.add_argument(
        "--stage",
        type=str,
        choices=[
            "setup",
            "data",
            "preprocess",
            "train",
            "evaluate",
            "dashboard",
        ],
        default="setup",
        help="Pipeline stage to execute.",
    )

    args = parser.parse_args()

    print_banner()

    if args.check_data:
        report_data_status()

    elif args.analyze:
        execute_analysis(
            args.analyze,
            sample_size=args.sample_size
        )

    elif args.run_baseline:
        execute_baseline_experiment(
            args.run_baseline,
            sample_size=args.sample_size
        )

    elif args.run_imbalance:
        run_imbalance_experiment()

    elif args.run_hybrid:
        run_hybrid_workflow()

    elif args.run_interpretability:
        run_interpretability()

    elif args.test_fixture:
        run_fixture_analysis(args.test_fixture)

    elif args.check or args.stage == "setup":
        print("\nStage: Project Setup & Experiment Execution")

        is_ready = check_environment()

        if not is_ready:
            print(
                "Action item: Activate your virtual environment "
                "and install missing requirements."
            )
        else:
            print("Project setup verified. Ready for experiment execution.")

    else:
        print(f"\nStage requested: {args.stage}")
        print("Use the available CLI commands to execute experiments.")


if __name__ == "__main__":
    main()
