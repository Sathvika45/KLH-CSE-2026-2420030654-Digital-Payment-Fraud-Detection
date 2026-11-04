
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "outputs" / "reports"

st.set_page_config(
    page_title="Digital Payment Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ Digital Payment Fraud Detection")
st.caption(
    "Comparative Analysis of Fraud Detection Techniques in Digital Payment Systems"
)


@st.cache_data
def load_csv(filename):
    path = REPORTS / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read {filename}: {exc}")
        return pd.DataFrame()


def canonical_dataset(value):
    aliases = {
        "credit_card": "Credit Card",
        "credit card": "Credit Card",
        "credit card fraud detection": "Credit Card",
        "ecommerce": "E-commerce",
        "e-commerce": "E-commerce",
        "fraud e-commerce": "E-commerce",
        "paysim": "PaySim",
        "paysim mobile money simulation": "PaySim",
    }
    text = str(value).strip()
    return aliases.get(text.lower(), text)


def dataset_column(df):
    for col in ("dataset", "dataset_name", "dataset_id"):
        if col in df.columns:
            return col
    return None


def filter_dataset(df, selected):
    if df.empty or selected == "All Datasets":
        return df.copy()
    col = dataset_column(df)
    if col is None:
        return pd.DataFrame()
    return df[df[col].map(canonical_dataset) == selected].copy()


def first_column(df, candidates):
    return next((col for col in candidates if col in df.columns), None)


baseline = load_csv("baseline_comparison_all_datasets.csv")
imbalance = load_csv("imbalance_comparison_summary.csv")
stability = load_csv("stability_fold_results.csv")
feature_importance = load_csv("random_forest_feature_importance.csv")
shap_importance = load_csv("random_forest_shap_importance.csv")
legacy_ablation = load_csv("hybrid_ablation_results.csv")
legacy_soft_voting = load_csv("hybrid_soft_voting_results.csv")


# SIDEBAR FILTERS
st.sidebar.header("Dashboard Filters")

found_datasets = set()
base_dataset_col = dataset_column(baseline)

if base_dataset_col:
    found_datasets.update(
        baseline[base_dataset_col].dropna().map(canonical_dataset).tolist()
    )

hybrid_files = {
    "Credit Card": "hybrid_credit_card_results.csv",
    "E-commerce": "hybrid_ecommerce_results.csv",
    "PaySim": "hybrid_paysim_results.csv",
}

for label, filename in hybrid_files.items():
    if (REPORTS / filename).exists():
        found_datasets.add(label)

order = ["Credit Card", "E-commerce", "PaySim"]

dataset_options = ["All Datasets"] + [
    x for x in order if x in found_datasets
]

dataset_options += sorted(found_datasets - set(order))

selected_dataset = st.sidebar.selectbox(
    "Select Dataset", dataset_options
)

threshold_type = st.sidebar.radio(
    "Evaluation Threshold",
    ["Tuned Threshold", "Default Threshold"]
)

suffix = "tuned" if threshold_type == "Tuned Threshold" else "default"

st.sidebar.markdown("---")
st.sidebar.caption(
    "Charts use saved experiment CSV files. Dashboard viewing does not retrain models."
)


# OVERVIEW
st.header("1. Model Performance Overview")

overview = filter_dataset(baseline, selected_dataset)

if overview.empty:
    st.info("Baseline overview is unavailable for this selection.")
else:
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Baseline Models Evaluated", len(overview))

    for col, label, metric_col in [
        (c2, "Best F1 Score", f"test_{suffix}_f1"),
        (c3, "Best PR-AUC", f"test_{suffix}_ap"),
        (c4, "Best ROC-AUC", f"test_{suffix}_roc_auc"),
    ]:
        if metric_col in overview.columns and overview[metric_col].notna().any():
            row = overview.loc[overview[metric_col].idxmax()]
            col.metric(
                label,
                f"{row[metric_col]:.4f}",
                str(row.get("display_name", "")),
            )
        else:
            col.metric(label, "N/A")


# FIVE DASHBOARD TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Baseline Comparison",
    "Imbalance Analysis",
    "Hybrid Analysis",
    "Stability Analysis",
    "Interpretability",
])


# TAB 1 - BASELINE COMPARISON
with tab1:
    st.subheader("Baseline Model Comparison")

    data = filter_dataset(baseline, selected_dataset)

    if data.empty:
        st.warning("No baseline comparison results are available.")
    else:
        metric_map = {
            "F1 Score": f"test_{suffix}_f1",
            "Precision": f"test_{suffix}_precision",
            "Recall": f"test_{suffix}_recall",
            "PR-AUC": f"test_{suffix}_ap",
            "ROC-AUC": f"test_{suffix}_roc_auc",
            "MCC": f"test_{suffix}_mcc",
        }

        metric_map = {
            k: v for k, v in metric_map.items()
            if v in data.columns
        }

        if metric_map:
            metric_label = st.selectbox(
                "Choose Performance Metric",
                list(metric_map),
                key="baseline_metric",
            )

            metric_col = metric_map[metric_label]

            name_col = first_column(
                data, ["display_name", "model"]
            )

            plot_data = data.dropna(subset=[metric_col])

            if name_col and not plot_data.empty:
                fig = px.bar(
                    plot_data.sort_values(
                        metric_col, ascending=False
                    ),
                    x=name_col,
                    y=metric_col,
                    color=name_col,
                    text_auto=".4f",
                    title=f"{metric_label} Comparison ({threshold_type})",
                )

                fig.update_layout(
                    showlegend=False,
                    xaxis_tickangle=-25
                )

                st.plotly_chart(
                    fig, use_container_width=True
                )

        time_cols = [
            c for c in ["train_time_sec", "test_ms_per_1k"]
            if c in data.columns
        ]

        name_col = first_column(
            data, ["display_name", "model"]
        )

        if time_cols and name_col:
            st.subheader("Training and Inference Time")

            melted = data[[name_col] + time_cols].melt(
                id_vars=name_col,
                var_name="Measurement",
                value_name="Time"
            )

            fig = px.bar(
                melted,
                x=name_col,
                y="Time",
                color="Measurement",
                barmode="group",
                title="Model Training and Inference Time"
            )

            st.plotly_chart(
                fig, use_container_width=True
            )

        st.subheader("Detailed Baseline Results")
        st.dataframe(data, use_container_width=True)


# TAB 2 - IMBALANCE ANALYSIS
with tab2:
    st.subheader("Class Imbalance Handling Techniques")

    data = filter_dataset(imbalance, selected_dataset)

    if data.empty:
        st.info("No imbalance results found for this dataset.")
    else:
        if "experiment_status" in data.columns:
            status = data["experiment_status"].astype(str).str.lower()

            completed = data[
                status.isin(["completed", "nan", ""])
            ]
        else:
            completed = data.copy()

        metric_map = {
            "F1 Score": f"{suffix}_f1",
            "PR-AUC": f"{suffix}_pr_auc",
            "Precision": f"{suffix}_precision",
            "Recall": f"{suffix}_recall",
            "ROC-AUC": f"{suffix}_roc_auc",
            "MCC": f"{suffix}_mcc",
        }

        metric_map = {
            k: v for k, v in metric_map.items()
            if v in completed.columns
        }

        if metric_map:
            label = st.selectbox(
                "Choose Imbalance Metric",
                list(metric_map),
                key="imbalance_metric"
            )

            metric_col = metric_map[label]

            x_col = first_column(
                completed, ["strategy", "strategy_type"]
            )

            color_col = (
                "strategy_type"
                if "strategy_type" in completed.columns
                else x_col
            )

            plot_data = completed.dropna(
                subset=[metric_col]
            )

            if x_col and not plot_data.empty:
                fig = px.bar(
                    plot_data.sort_values(
                        metric_col, ascending=False
                    ),
                    x=x_col,
                    y=metric_col,
                    color=color_col,
                    text_auto=".4f",
                    title=f"{label} Across Imbalance Strategies"
                )

                fig.update_layout(xaxis_tickangle=-30)

                st.plotly_chart(
                    fig, use_container_width=True
                )
        else:
            st.info(
                "No compatible metric columns found in the imbalance report."
            )

        st.subheader("Imbalance Experiment Results")
        st.dataframe(data, use_container_width=True)

        if "experiment_status" in data.columns:
            skipped = data[
                data["experiment_status"].astype(str).str.lower().isin(
                    ["not_applicable", "not applicable", "skipped"]
                )
            ]

            if not skipped.empty:
                st.caption(
                    f"{len(skipped)} row(s) are marked not applicable or skipped "
                    "and are excluded from metric charts."
                )


# TAB 3 - HYBRID ANALYSIS
with tab3:
    st.subheader("Proposed Hybrid Model Evaluation")

    st.markdown(
        "**Architecture:** Random Forest and LightGBM are the base learners; "
        "Logistic Regression is the stacking meta-classifier."
    )

    st.code(
        """Input Transaction Data
          |
    +-----+------+
    |            |
Random Forest  LightGBM
    |            |
    +-----+------+
          |
 Logistic Regression
    (Meta-Learner)
          |
 Final Fraud Prediction""",
        language="text",
    )

    st.divider()

    hybrid_data = pd.DataFrame()

    if selected_dataset == "All Datasets":
        frames = []

        for label, filename in hybrid_files.items():
            frame = load_csv(filename)

            if not frame.empty:
                frame = frame.copy()

                if dataset_column(frame) is None:
                    frame.insert(0, "dataset", label)

                frames.append(frame)

        if frames:
            hybrid_data = pd.concat(
                frames, ignore_index=True, sort=False
            )

    else:
        filename = hybrid_files.get(selected_dataset)

        if filename:
            hybrid_data = load_csv(filename)

    # Credit Card legacy reports are used only as Credit Card data.
    if hybrid_data.empty and selected_dataset in (
        "Credit Card", "All Datasets"
    ):
        if not legacy_ablation.empty:
            hybrid_data = legacy_ablation.copy()

            if dataset_column(hybrid_data) is None:
                hybrid_data.insert(
                    0, "dataset", "Credit Card"
                )

    if hybrid_data.empty:
        st.info(
            "No saved hybrid result file was found for this selection. "
            "Expected files: hybrid_credit_card_results.csv, "
            "hybrid_ecommerce_results.csv, hybrid_paysim_results.csv."
        )

    else:
        st.markdown("#### Saved Hybrid Experiment Results")

        metric_candidates = {
            "Precision": ["tuned_precision", "precision"],
            "Recall": ["tuned_recall", "recall"],
            "F1 Score": ["tuned_f1", "f1"],
            "PR-AUC": ["tuned_pr_auc", "pr_auc", "average_precision"],
            "ROC-AUC": ["tuned_roc_auc", "roc_auc"],
            "MCC": ["tuned_mcc", "mcc"],
        }

        metric_map = {
            label: col
            for label, candidates in metric_candidates.items()
            if (col := first_column(hybrid_data, candidates))
        }

        if metric_map:
            metric_label = st.selectbox(
                "Choose Hybrid Metric",
                list(metric_map),
                key="hybrid_metric"
            )

            metric_col = metric_map[metric_label]

            x_col = first_column(
                hybrid_data,
                ["model", "evaluation", "experiment", "variant"]
            )

            color_col = dataset_column(hybrid_data)

            plot_data = hybrid_data.dropna(
                subset=[metric_col]
            )

            if x_col and not plot_data.empty:
                fig = px.bar(
                    plot_data,
                    x=x_col,
                    y=metric_col,
                    color=(
                        color_col
                        if color_col and color_col != x_col
                        else None
                    ),
                    barmode="group",
                    text_auto=".4f",
                    title=f"Hybrid {metric_label} Comparison"
                )

                fig.update_layout(xaxis_tickangle=-25)

                st.plotly_chart(
                    fig, use_container_width=True
                )

            else:
                st.info(
                    "Hybrid results are loaded, but no model/evaluation "
                    "column is available for a metric chart."
                )

        st.dataframe(
            hybrid_data, use_container_width=True
        )

    if (
        selected_dataset in ("Credit Card", "All Datasets")
        and not legacy_soft_voting.empty
    ):
        st.markdown(
            "#### Legacy Soft-Voting Results (Credit Card)"
        )
        st.dataframe(
            legacy_soft_voting, use_container_width=True
        )

    st.caption(
        "These are saved experiment results. This tab does not retrain models "
        "or generate new predictions."
    )


# TAB 4 - STABILITY ANALYSIS
with tab4:
    st.subheader("Five-Fold Stability Analysis")

    data = filter_dataset(stability, selected_dataset)

    if data.empty:
        st.info("No stability results found for this dataset.")
    else:
        metrics = [
            c for c in [
                "precision",
                "recall",
                "f1",
                "average_precision",
                "roc_auc",
                "training_time_sec"
            ]
            if c in data.columns
        ]

        if "model" in data.columns and metrics:
            summary = (
                data.groupby("model")[metrics]
                .agg(["mean", "std"])
                .round(4)
            )

            st.markdown(
                "#### Mean and Standard Deviation Across Folds"
            )

            st.dataframe(
                summary, use_container_width=True
            )

        chart_metrics = [
            c for c in [
                "precision",
                "recall",
                "f1",
                "average_precision",
                "roc_auc"
            ]
            if c in data.columns
        ]

        if chart_metrics and {
            "fold", "model"
        }.issubset(data.columns):

            metric = st.selectbox(
                "Choose Stability Metric",
                chart_metrics,
                key="stability_metric"
            )

            fig = px.line(
                data,
                x="fold",
                y=metric,
                color="model",
                markers=True,
                title=f"{metric.replace('_', ' ').title()} Across Folds"
            )

            st.plotly_chart(
                fig, use_container_width=True
            )

        if {
            "fold", "model", "training_time_sec"
        }.issubset(data.columns):

            fig = px.line(
                data,
                x="fold",
                y="training_time_sec",
                color="model",
                markers=True,
                title="Training Time Across Folds"
            )

            st.plotly_chart(
                fig, use_container_width=True
            )

        st.markdown("#### Fold-Level Results")

        st.dataframe(
            data, use_container_width=True
        )


# TAB 5 - INTERPRETABILITY
with tab5:
    st.subheader("Model Interpretability")

    if selected_dataset not in (
        "Credit Card", "All Datasets"
    ):
        st.info(
            "The currently confirmed feature-importance and SHAP reports "
            "are for Random Forest on the Credit Card dataset. "
            "No interpretability result is claimed for this dataset here."
        )

    else:
        st.caption(
            "Available reports: Random Forest feature importance and "
            "mean absolute SHAP importance for Credit Card."
        )

        if not feature_importance.empty and {
            "feature", "importance"
        }.issubset(feature_importance.columns):

            st.markdown(
                "#### Random Forest Feature Importance"
            )

            features = (
                feature_importance
                .nlargest(15, "importance")
                .sort_values("importance")
            )

            fig = px.bar(
                features,
                x="importance",
                y="feature",
                orientation="h",
                title="Top 15 Random Forest Features"
            )

            st.plotly_chart(
                fig, use_container_width=True
            )

            st.dataframe(
                feature_importance, use_container_width=True
            )

        else:
            st.info(
                "Random Forest feature-importance report is unavailable."
            )

        st.divider()

        if not shap_importance.empty and {
            "feature", "mean_absolute_shap"
        }.issubset(shap_importance.columns):

            st.markdown(
                "#### SHAP Feature Importance"
            )

            shap_data = (
                shap_importance
                .nlargest(15, "mean_absolute_shap")
                .sort_values("mean_absolute_shap")
            )

            fig = px.bar(
                shap_data,
                x="mean_absolute_shap",
                y="feature",
                orientation="h",
                title="Top 15 Features by Mean Absolute SHAP"
            )

            st.plotly_chart(
                fig, use_container_width=True
            )

            st.dataframe(
                shap_importance, use_container_width=True
            )

        else:
            st.info(
                "SHAP importance report is unavailable."
            )


# FOOTER
st.divider()

st.caption(
    "Digital Payment Fraud Detection | Experimental Results Dashboard"
)
