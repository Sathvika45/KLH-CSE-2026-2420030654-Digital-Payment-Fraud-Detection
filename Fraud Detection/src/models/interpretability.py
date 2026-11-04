
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import shap

from src.preprocessing.features import engineer_features
from src.preprocessing.split import (
    prepare_data,
    create_stratified_splits
)
from src.models.baselines import get_baseline_model
from src.models.workflow import build_baseline_pipeline


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
REPORT_DIR = ROOT / "outputs" / "reports"
FIGURE_DIR = ROOT / "outputs" / "figures"


def run_interpretability():

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Credit Card Fraud dataset...")
    df = pd.read_csv(DATA_PATH)

    print("Engineering features...")
    df = engineer_features(df, "credit_card")

    print("Preparing data...")
    prepared_df, prep_summary = prepare_data(
        df,
        target_column="Class",
        drop_duplicates=True
    )

    print("Creating stratified train/validation/test splits...")
    splits = create_stratified_splits(
        prepared_df,
        target_column="Class",
        random_state=42
    )

    # Access SplitArtifacts using attributes
    X_train = splits.X_train
    y_train = splits.y_train
    X_test = splits.X_test
    y_test = splits.y_test

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    print("Building Random Forest pipeline...")
    model = get_baseline_model(
        "random_forest",
        random_state=42
    )

    pipeline = build_baseline_pipeline(X_train, model)

    print("Training Random Forest...")
    pipeline.fit(X_train, y_train)

    print("Extracting feature importance...")
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    importance_values = classifier.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance_values
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )
    
    # SHAP explainability using TreeSHAP
    print("Preparing data for SHAP analysis...")

    # Include all fraud cases and 200 randomly selected legitimate cases
    fraud_indices = y_test[y_test == 1].index
    legitimate_indices = y_test[y_test == 0].sample(
        n=min(200, int((y_test == 0).sum())),
        random_state=42
    ).index

    shap_indices = fraud_indices.union(legitimate_indices)
    X_shap = X_test.loc[shap_indices]

    # Apply the already-fitted preprocessing pipeline
    X_shap_transformed = preprocessor.transform(X_shap)

    print(f"SHAP samples: {len(X_shap_transformed)}")

    print("Calculating TreeSHAP values...")
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_shap_transformed)

    # Extract SHAP values for the fraud class
    if isinstance(shap_values, list):
        fraud_shap_values = shap_values[1]
    elif len(shap_values.shape) == 3:
        fraud_shap_values = shap_values[:, :, 1]
    else:
        fraud_shap_values = shap_values

    # Calculate mean absolute SHAP importance
    shap_importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_absolute_shap": abs(fraud_shap_values).mean(axis=0)
    }).sort_values(
        by="mean_absolute_shap",
        ascending=False
    )

    shap_csv_path = REPORT_DIR / "random_forest_shap_importance.csv"
    shap_importance_df.to_csv(shap_csv_path, index=False)

    # Generate SHAP summary plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        fraud_shap_values,
        X_shap_transformed,
        feature_names=feature_names,
        max_display=15,
        show=False
    )
    plt.tight_layout()

    shap_figure_path = FIGURE_DIR / "random_forest_shap_summary.png"
    plt.savefig(shap_figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"SHAP importance CSV: {shap_csv_path}")
    print(f"SHAP summary graph: {shap_figure_path}")


    # Save feature importance CSV
    csv_path = REPORT_DIR / "random_forest_feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)

    # Plot top 15 features
    top_features = importance_df.head(15).sort_values(
        by="importance",
        ascending=True
    )

    plt.figure(figsize=(10, 7))
    plt.barh(
        top_features["feature"],
        top_features["importance"]
    )
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("Random Forest - Top 15 Important Features")
    plt.tight_layout()

    figure_path = FIGURE_DIR / "random_forest_feature_importance.png"
    plt.savefig(figure_path, dpi=300)
    plt.close()

    # Generate Markdown report
    report_path = REPORT_DIR / "interpretability_report.md"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("# Model Interpretability Report\n\n")

        file.write("## Method\n")
        file.write(
            "Random Forest impurity-based feature importance "
            "was used to identify the features contributing most "
            "to the model's predictions.\n\n"
        )

        file.write("## Top 15 Important Features\n\n")
        file.write(
            importance_df.head(15).to_markdown(index=False)
        )
        file.write("\n\n## SHAP Analysis\n\n")
        file.write(
            "TreeSHAP was used to explain the Random Forest model's "
            "fraud-class predictions. All fraud cases and up to 200 "
            "randomly selected legitimate test cases were included.\n\n"
        )
        file.write("### Top 15 SHAP Features\n\n")
        file.write(
            shap_importance_df.head(15).to_markdown(index=False)
        )
        file.write(
            "\n\nThe SHAP summary plot shows the contribution of "
            "individual features to the model output. SHAP values "
            "describe model behaviour and do not establish causation.\n"
        )

        file.write(
            "\n\n## Limitation\n"
            "Feature importance indicates how the model uses "
            "features. It does not establish causation. "
            "Correlated features can affect importance values.\n"
        )

    print("\nInterpretability experiment completed!")
    print(f"CSV: {csv_path}")
    print(f"Graph: {figure_path}")
    print(f"Report: {report_path}")

    print("\nTop 10 important features:")
    print(importance_df.head(10).to_string(index=False))


if __name__ == "__main__":
    run_interpretability()
