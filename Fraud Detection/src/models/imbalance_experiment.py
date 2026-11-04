
from pathlib import Path
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.data.loader import load_dataset
from src.preprocessing.features import engineer_features
from src.preprocessing.split import prepare_data, create_stratified_splits
from src.preprocessing.imbalance import (
    ImbalanceStrategy,
    get_imbalance_pipeline,
    get_candidate_resampling_ratios,
)
from src.evaluation.metrics import (
    compute_evaluation_metrics,
    find_optimal_threshold,
)


REPORT_DIR = Path("outputs/reports")
RANDOM_STATE = 42

# PaySim is sampled to match the existing baseline experiment.
DATASET_CONFIG = {
    "paysim": {
        "name": "PaySim",
        "target": "isFraud",
        "sample_size": 100000,
    },
    "ecommerce": {
        "name": "E-commerce Fraud",
        "target": "class",
        "sample_size": None,
    },
}


def make_model(class_weight=None):
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=5,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_preprocessor(X_train):
    """Create numeric and categorical preprocessing fitted on training data."""

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

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def build_experiment_strategies():
    """Create control experiments and ratio-based resampling experiments."""

    strategies = [
        {
            "name": "original",
            "strategy": ImbalanceStrategy.ORIGINAL,
            "class_weight": None,
            "ratio": None,
        },
        {
            "name": "class_weighted",
            "strategy": ImbalanceStrategy.CLASS_WEIGHTED,
            "class_weight": "balanced",
            "ratio": None,
        },
    ]

    ratios = get_candidate_resampling_ratios()

    for ratio_name, ratio_value in ratios.items():
        strategies.append({
            "name": f"smote_{ratio_name.replace(':', '_')}",
            "strategy": ImbalanceStrategy.SMOTE,
            "class_weight": None,
            "ratio": ratio_value,
        })

        strategies.append({
            "name": f"undersampling_{ratio_name.replace(':', '_')}",
            "strategy": ImbalanceStrategy.UNDERSAMPLING,
            "class_weight": None,
            "ratio": ratio_value,
        })

    return strategies


def run_dataset_experiments(dataset_id):
    """Run all imbalance experiments for one dataset."""

    config = DATASET_CONFIG[dataset_id]
    target_column = config["target"]

    print("\n" + "=" * 65)
    print(f"DATASET: {config['name']}")
    print("=" * 65)

    print("Loading dataset...")

    load_result = load_dataset(
        dataset_id=dataset_id,
        sample_size=config["sample_size"],
        random_state=RANDOM_STATE,
    )

    df = load_result.df

    print(f"Source records: {load_result.total_records:,}")
    print(f"Loaded records: {load_result.loaded_records:,}")
    print(f"Sampled: {load_result.is_sampled}")

    # Deduplicate before splitting.
    clean_df, audit = prepare_data(
        df,
        target_column=target_column,
        drop_duplicates=True,
    )

    print(f"Rows after duplicate removal: {len(clean_df):,}")
    print(f"Duplicates removed: {audit.dropped_duplicates:,}")

    # Separate target before feature engineering.
    y = clean_df[target_column].copy()
    X = clean_df.drop(columns=[target_column]).copy()

    X = engineer_features(X, dataset_id)

    # Recombine features and target for the existing split function.
    prepared_df = X.copy()
    prepared_df[target_column] = y

    splits = create_stratified_splits(
        prepared_df,
        target_column=target_column,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        random_state=RANDOM_STATE,
    )

    X_train, y_train = splits.X_train, splits.y_train
    X_val, y_val = splits.X_val, splits.y_val
    X_test, y_test = splits.X_test, splits.y_test

    print("\nTraining class distribution:")
    print(y_train.value_counts().to_dict())

    print("\nSplit sizes:")
    print(f"Training:   {len(X_train):,}")
    print(f"Validation: {len(X_val):,}")
    print(f"Test:       {len(X_test):,}")

    # This preprocessor is passed into the imbalanced-learn pipeline.
    # It is fitted only when pipeline.fit(X_train, y_train) is called.
    preprocessor = build_preprocessor(X_train)

    strategies = build_experiment_strategies()
    results = []
    current_ratio = float((y_train == 1).sum() / (y_train == 0).sum())

    print(f"\nTotal experiments planned: {len(strategies)}")

    for experiment_number, experiment in enumerate(strategies, start=1):
        name = experiment["name"]
        strategy = experiment["strategy"]
        class_weight = experiment["class_weight"]
        ratio = experiment["ratio"]
                # SMOTE and RandomUnderSampler can only increase
        # the minority-to-majority ratio.
        if ratio is not None and ratio <= current_ratio:
            print(
                f"Skipping {name}: requested ratio {ratio:.4f} "
                f"is not greater than natural training ratio "
                f"{current_ratio:.4f}."
            )

            results.append({
                "dataset_id": dataset_id,
                "dataset_name": config["name"],
                "strategy": name,
                "strategy_type": strategy.value,
                "target_sampling_ratio": f"1:{int(round(1 / ratio))}",
                "experiment_status": "Not applicable",
                "skip_reason": (
                    "Requested sampling ratio is not greater "
                    "than the natural training ratio."
                ),
            })
            continue

        print(
            f"\n[{experiment_number}/{len(strategies)}] "
            f"{config['name']} - {name}"
        )

        estimator = make_model(class_weight=class_weight)

        pipeline = get_imbalance_pipeline(
            strategy=strategy,
            base_estimator=estimator,
            scaler=preprocessor,
            random_state=RANDOM_STATE,
            smote_sampling_strategy=(
                ratio if ratio is not None else 0.10
            ),
            under_sampling_strategy=(
                ratio if ratio is not None else 0.10
            ),
        )

        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start

        # Select threshold using validation data only.
        val_probabilities = pipeline.predict_proba(X_val)[:, 1]

        tuned_threshold, _, _ = find_optimal_threshold(
            y_val,
            val_probabilities,
            metric="f1",
        )

        # Test data is not resampled or used for threshold selection.
        test_probabilities = pipeline.predict_proba(X_test)[:, 1]

        default_metrics = compute_evaluation_metrics(
            y_test,
            test_probabilities,
            threshold=0.50,
        )

        tuned_metrics = compute_evaluation_metrics(
            y_test,
            test_probabilities,
            threshold=tuned_threshold,
        )

        result = {
            "dataset_id": dataset_id,
            "dataset_name": config["name"],
            "experiment_status": "Completed",
            "source_records": load_result.total_records,
            "loaded_records": load_result.loaded_records,
            "is_sampled": load_result.is_sampled,
            "sampling_method": load_result.sampling_method,
            "random_state": RANDOM_STATE,
            "duplicates_removed": audit.dropped_duplicates,
            "train_records": len(X_train),
            "validation_records": len(X_val),
            "test_records": len(X_test),
            "train_fraud_count": int((y_train == 1).sum()),
            "validation_fraud_count": int((y_val == 1).sum()),
            "test_fraud_count": int((y_test == 1).sum()),
            "strategy": name,
            "strategy_type": strategy.value,
            "target_sampling_ratio": (
                f"1:{int(round(1 / ratio))}"
                if ratio is not None
                else "Not applicable"
            ),
            "train_time_sec": round(train_seconds, 3),
            "frozen_validation_threshold": round(
                float(tuned_threshold), 6
            ),
        }

        for key, value in default_metrics.items():
            if isinstance(value, (int, float)):
                result[f"default_{key}"] = value

        for key, value in tuned_metrics.items():
            if isinstance(value, (int, float)):
                result[f"tuned_{key}"] = value

        results.append(result)

        print(
            f"Finished {name}: "
            f"AP={default_metrics.get('pr_auc', float('nan')):.4f}, "
            f"F1={default_metrics.get('f1', float('nan')):.4f}, "
            f"training={train_seconds:.1f}s"
        )

    return results


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = REPORT_DIR / "imbalance_comparison_summary.csv"
    md_path = REPORT_DIR / "imbalance_comparison_report.md"

    # Preserve the already completed Credit Card experiments.
    if not csv_path.exists():
        raise FileNotFoundError(
            "Existing Credit Card imbalance results were not found. "
            "Expected: outputs/reports/imbalance_comparison_summary.csv"
        )

    print("Reading completed Credit Card experiments...")
    credit_results = pd.read_csv(csv_path)

    if "dataset_id" in credit_results.columns:
        credit_results = credit_results[
            credit_results["dataset_id"] == "credit_card"
        ].copy()

    if credit_results.empty:
        raise ValueError(
            "No completed Credit Card results were found in the existing CSV."
        )

    credit_results["dataset_id"] = "credit_card"
    credit_results["dataset_name"] = "Credit Card Fraud"

    print(
        f"Preserving {len(credit_results)} existing "
        "Credit Card experiments. They will not be rerun."
    )

    new_results = []

    # Run the remaining two datasets.
    for dataset_id in ["paysim", "ecommerce"]:
        dataset_results = run_dataset_experiments(dataset_id)
        new_results.extend(dataset_results)

    new_results_df = pd.DataFrame(new_results)

    # Combine existing Credit Card results with new results.
    combined_df = pd.concat(
        [credit_results, new_results_df],
        ignore_index=True,
        sort=False,
    )

    combined_df.to_csv(csv_path, index=False)

    report_lines = [
        "# Comparative Imbalance Handling Experiments",
        "",
        "## Experimental scope",
        "",
        "This report combines the previously completed Credit Card "
        "experiments with newly executed PaySim and E-commerce experiments.",
        "",
        "Random Forest was evaluated using:",
        "- Original imbalanced distribution.",
        "- Cost-sensitive class weighting.",
        "- SMOTE at five target sampling ratios.",
        "- Controlled random undersampling at five target sampling ratios.",
        "",
        "The target sampling ratios represent the minority-to-majority "
        "class ratio in the resampled training data.",
        "",
        "## Dataset handling",
        "",
        "- Credit Card: previously completed results were reused.",
        "- PaySim: stratified 100,000-record sample, matching the baseline.",
        "- E-commerce: full dataset.",
        "- Each new dataset uses stratified 60/20/20 training, validation, "
        "and test partitions with random seed 42.",
        "- Duplicate removal is performed before splitting.",
        "- Numeric and categorical preprocessing is fitted on training data.",
        "- Resampling is performed only within the training pipeline.",
        "- Validation data selects the decision threshold.",
        "- Test data retains its natural class distribution.",
        "",
        "## Results",
        "",
        combined_df.to_markdown(index=False),
        "",
        "## Interpretation notes",
        "",
        "- Compare AP, precision, recall, F1, MCC, and false-positive rate together.",
        "- The validation-selected threshold is frozen before test evaluation.",
        "- AP is the Average Precision score returned by the evaluator, "
        "not trapezoidal PR-AUC.",
        "- PaySim results represent a 100,000-record sample, not the full dataset.",
        "- Results across datasets should be interpreted in the context "
        "of their different feature spaces and class distributions.",
        "- No single resampling strategy should be declared universally "
        "best based on these experiments alone.",
        "",
    ]

    md_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("ALL DATASET EXPERIMENTS COMPLETED")
    print("=" * 65)
    print(f"Credit Card experiments preserved: {len(credit_results)}")
    print(f"New experiments completed: {len(new_results_df)}")
    print(f"Total consolidated experiments: {len(combined_df)}")
    print(f"CSV: {csv_path.resolve()}")
    print(f"Report: {md_path.resolve()}")


if __name__ == "__main__":
    main()
