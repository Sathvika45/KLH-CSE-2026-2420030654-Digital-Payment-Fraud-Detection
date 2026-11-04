
from pathlib import Path
import time

import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)

from src.data.loader import load_dataset
from src.preprocessing.features import engineer_features
from src.preprocessing.split import prepare_data
from src.models.baselines import get_baseline_model
from src.models.workflow import build_baseline_pipeline


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "outputs" / "reports"
RESULTS_PATH = REPORT_DIR / "stability_fold_results.csv"
SUMMARY_PATH = REPORT_DIR / "stability_summary.csv"

DATASETS = {
    "paysim": 100_000,
    "ecommerce": None,
}

N_SPLITS = 5
RANDOM_STATE = 42


def save_results(results):
    """Save fold-level results and mean/std summary."""
    results_df = pd.DataFrame(results)

    results_df.to_csv(RESULTS_PATH, index=False)

    summary = (
        results_df.groupby(["dataset", "model"])
        .agg({
            "precision": ["mean", "std"],
            "recall": ["mean", "std"],
            "f1": ["mean", "std"],
            "average_precision": ["mean", "std"],
            "roc_auc": ["mean", "std"],
            "training_time_sec": ["mean", "std"],
        })
    )

    summary.to_csv(SUMMARY_PATH)

    print("\nCurrent stability summary:")
    print(summary.to_string())


def build_stacking_model(X_train, fold_number):
    """RF + LightGBM base models with Logistic Regression meta-model."""

    rf = get_baseline_model(
        "random_forest",
        random_state=RANDOM_STATE + fold_number,
    )

    lgbm = get_baseline_model(
        "lightgbm",
        random_state=RANDOM_STATE + fold_number,
    )

    rf_pipeline = build_baseline_pipeline(X_train, rf)
    lgbm_pipeline = build_baseline_pipeline(X_train, lgbm)

    return StackingClassifier(
        estimators=[
            ("random_forest", rf_pipeline),
            ("lightgbm", lgbm_pipeline),
        ],
        final_estimator=LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        stack_method="predict_proba",
        cv=3,
        n_jobs=1,
        passthrough=False,
    )


def prepare_dataset(dataset_id, sample_size):
    """Load and prepare a dataset without losing valid transactions."""

    loaded = load_dataset(
        dataset_id=dataset_id,
        sample_size=sample_size,
        random_state=RANDOM_STATE,
    )

    df = loaded.df.copy()
    target_column = loaded.dataset_config.target_column

    print(f"\nSource records: {loaded.total_records:,}")
    print(f"Records loaded: {loaded.loaded_records:,}")
    print(f"Target column: {target_column}")
    print(f"Sampled: {loaded.is_sampled}")

    if loaded.unmatched_ip_count is not None:
        print(
            f"Unmatched IP addresses: "
            f"{loaded.unmatched_ip_count:,}"
        )

    print(
        "Original class counts:",
        df[target_column].value_counts().to_dict(),
    )

    if dataset_id == "ecommerce":
        # Deduplicate raw records BEFORE dropping identifiers
        # and timestamps through feature engineering.
        prepared_raw, prep_summary = prepare_data(
            df,
            target_column=target_column,
            drop_duplicates=True,
        )

        print("\nE-COMMERCE RAW-DATA AUDIT")
        print(prep_summary.to_dict())

        # Engineer only the feature columns.
        X = prepared_raw.drop(columns=[target_column])
        X = engineer_features(X, dataset_id)

        # Restore the original target without changing row order.
        X = X.reset_index(drop=True)
        y = prepared_raw[target_column].reset_index(drop=True)

        prepared_df = X.copy()
        prepared_df[target_column] = y

        print("\nE-COMMERCE POST-FEATURE-ENGINEERING AUDIT")
        print(f"Prepared records: {len(prepared_df):,}")
        print(
            "Class counts:",
            prepared_df[target_column].value_counts().to_dict(),
        )

    else:
        # Preserve the existing PaySim preprocessing protocol.
        df = engineer_features(df, dataset_id)

        prepared_df, prep_summary = prepare_data(
            df,
            target_column=target_column,
            drop_duplicates=True,
        )

        print("\nPREPROCESSING AUDIT")
        print(prep_summary.to_dict())

    X = prepared_df.drop(columns=[target_column])
    y = prepared_df[target_column]

    print(f"\nFinal records: {len(X):,}")
    print(f"Final fraud records: {int(y.sum()):,}")
    print(f"Final legitimate records: {int((y == 0).sum()):,}")

    return X, y


def run_dataset_stability(dataset_id, sample_size, results):

    print(f"\n\n{'=' * 60}")
    print(f"STABILITY ANALYSIS: {dataset_id.upper()}")
    print(f"{'=' * 60}")

    X, y = prepare_dataset(dataset_id, sample_size)

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y), start=1
    ):

        print(
            f"\n========== {dataset_id.upper()} "
            f"FOLD {fold}/{N_SPLITS} =========="
        )

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        models = {
            "Random Forest": build_baseline_pipeline(
                X_train,
                get_baseline_model(
                    "random_forest",
                    random_state=RANDOM_STATE + fold,
                ),
            ),
            "Stacking Hybrid": build_stacking_model(
                X_train,
                fold,
            ),
        }

        for name, model in models.items():

            already_done = any(
                row["dataset"] == dataset_id
                and int(row["fold"]) == fold
                and row["model"] == name
                for row in results
            )

            if already_done:
                print(f"Skipping completed {name} fold {fold}.")
                continue

            print(f"\nTraining {name}...")

            start = time.perf_counter()
            model.fit(X_train, y_train)
            training_time = time.perf_counter() - start

            probabilities = model.predict_proba(X_test)[:, 1]
            predictions = (probabilities >= 0.5).astype(int)

            metrics = {
                "dataset": dataset_id,
                "fold": fold,
                "model": name,
                "precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "recall": recall_score(
                    y_test, predictions, zero_division=0
                ),
                "f1": f1_score(
                    y_test, predictions, zero_division=0
                ),
                "average_precision": average_precision_score(
                    y_test, probabilities
                ),
                "roc_auc": roc_auc_score(
                    y_test, probabilities
                ),
                "training_time_sec": training_time,
            }

            results.append(metrics)

            # Save after every completed model.
            save_results(results)

            print(
                f"Precision: {metrics['precision']:.4f} | "
                f"Recall: {metrics['recall']:.4f} | "
                f"F1: {metrics['f1']:.4f} | "
                f"AP: {metrics['average_precision']:.4f} | "
                f"ROC-AUC: {metrics['roc_auc']:.4f} | "
                f"Time: {training_time:.2f}s"
            )


def run_stability_analysis():

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Existing stability results not found: "
            f"{RESULTS_PATH}"
        )

    results_df = pd.read_csv(RESULTS_PATH)

    if "dataset" not in results_df.columns:
        results_df.insert(0, "dataset", "credit_card")

    # Preserve Credit Card and PaySim results.
    # Remove old E-commerce results because preprocessing is corrected.
    old_count = len(results_df)

    results_df = results_df[
        results_df["dataset"] != "ecommerce"
    ].copy()

    removed_count = old_count - len(results_df)

    results = results_df.to_dict("records")

    print("Existing results loaded.")
    print("Credit Card results will NOT be retrained.")
    print("Completed PaySim folds will be skipped.")
    print(
        f"Old E-commerce result rows removed: {removed_count}"
    )

    # Run PaySim and corrected E-commerce.
    for dataset_id, sample_size in DATASETS.items():

        run_dataset_stability(
            dataset_id,
            sample_size,
            results,
        )

    save_results(results)

    print("\nALL DATASET STABILITY ANALYSIS COMPLETED.")
    print(f"\nFold results: {RESULTS_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    run_stability_analysis()
