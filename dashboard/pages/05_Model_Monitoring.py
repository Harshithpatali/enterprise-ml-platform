from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Model Monitoring",
    page_icon="📊",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"

METADATA_PATH = MODELS_DIR / "model_metadata.json"
VALIDATION_RESULTS_PATH = MODELS_DIR / "validation_results.csv"
THRESHOLD_PATH = MODELS_DIR / "threshold_analysis.csv"
FEATURE_IMPORTANCE_PATH = MODELS_DIR / "feature_importance.csv"


# ============================================================
# HELPERS
# ============================================================

@st.cache_data
def load_json(path):
    if not path.exists():
        return None

    with open(path, "r") as f:
        return json.load(f)


@st.cache_data
def load_csv(path):
    if not path.exists():
        return None

    return pd.read_csv(path)


metadata = load_json(METADATA_PATH)
validation_df = load_csv(VALIDATION_RESULTS_PATH)
threshold_df = load_csv(THRESHOLD_PATH)
feature_importance_df = load_csv(FEATURE_IMPORTANCE_PATH)


# ============================================================
# HEADER
# ============================================================

st.title("📊 Model Monitoring")

st.caption(
    "Production monitoring and model governance for the "
    "Enterprise Churn Intelligence platform"
)

st.markdown(
    """
This dashboard monitors the currently deployed **V2 XGBoost champion
model** using the evaluation artifacts generated during model training.

The page separates:

- Model performance
- Threshold behavior
- Feature importance
- Model metadata
- Production readiness
- Monitoring limitations
"""
)


# ============================================================
# MODEL STATUS
# ============================================================

st.subheader("🚦 Production Model Status")

if metadata is not None:

    champion_name = metadata.get(
        "champion_model",
        "Unknown",
    )

    target = metadata.get(
        "target",
        "Unknown",
    )

    threshold = metadata.get(
        "threshold",
        None,
    )

    selection_metric = metadata.get(
        "selection_metric",
        metadata.get(
            "selection_criterion",
            "Unknown",
        ),
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Model",
            str(champion_name).upper(),
        )

    with col2:
        st.metric(
            "Target",
            target,
        )

    with col3:
        if threshold is not None:
            st.metric(
                "Decision Threshold",
                f"{float(threshold):.2f}",
            )
        else:
            st.metric(
                "Decision Threshold",
                "N/A",
            )

    with col4:
        st.metric(
            "Selection Metric",
            str(selection_metric),
        )

else:

    st.warning(
        "Model metadata file was not found."
    )


# ============================================================
# PERFORMANCE METRICS
# ============================================================

st.divider()

st.subheader("🎯 Model Performance")

test_metrics = {}

if metadata is not None:

    # The V2 metadata stores test metrics inside the test_metrics object.
    test_metrics = metadata.get(
        "test_metrics",
        {},
    )

    # Fallback in case metrics are stored at the root level.
    if not test_metrics:

        possible_metrics = [
            "roc_auc",
            "pr_auc",
            "precision",
            "recall",
            "f1",
            "accuracy",
        ]

        test_metrics = {
            key: metadata[key]
            for key in possible_metrics
            if key in metadata
        }


def get_metric(data, *names):

    for name in names:

        if name in data:
            return data[name]

    return None


roc_auc = get_metric(
    test_metrics,
    "roc_auc",
    "ROC-AUC",
    "roc_auc_score",
)

pr_auc = get_metric(
    test_metrics,
    "pr_auc",
    "PR-AUC",
    "average_precision",
)

precision = get_metric(
    test_metrics,
    "precision",
)

recall = get_metric(
    test_metrics,
    "recall",
)

f1 = get_metric(
    test_metrics,
    "f1",
    "f1_score",
)


col1, col2, col3, col4, col5 = st.columns(5)

with col1:

    st.metric(
        "ROC-AUC",
        f"{float(roc_auc):.4f}"
        if roc_auc is not None
        else "N/A",
    )

with col2:

    st.metric(
        "PR-AUC",
        f"{float(pr_auc):.4f}"
        if pr_auc is not None
        else "N/A",
    )

with col3:

    st.metric(
        "Precision",
        f"{float(precision):.4f}"
        if precision is not None
        else "N/A",
    )

with col4:

    st.metric(
        "Recall",
        f"{float(recall):.4f}"
        if recall is not None
        else "N/A",
    )

with col5:

    st.metric(
        "F1",
        f"{float(f1):.4f}"
        if f1 is not None
        else "N/A",
    )


# ============================================================
# VALIDATION MODEL COMPARISON
# ============================================================

st.divider()

st.subheader("🏆 Model Validation Comparison")

if validation_df is not None and not validation_df.empty:

    st.markdown(
        """
The validation comparison shows the candidate models evaluated during
the training pipeline before selecting the champion.
"""
    )

    st.dataframe(
        validation_df,
        use_container_width=True,
        hide_index=True,
    )

    # Identify likely metric columns
    model_column = None
    pr_column = None
    roc_column = None
    f1_column = None

    for column in validation_df.columns:

        lower = str(column).lower()

        if lower in ["model", "model_name", "algorithm"]:
            model_column = column

        elif "pr" in lower and "auc" in lower:
            pr_column = column

        elif "roc" in lower and "auc" in lower:
            roc_column = column

        elif lower == "f1" or "f1" in lower:
            f1_column = column

    if model_column is not None and pr_column is not None:

        chart_columns = [
            model_column,
            pr_column,
        ]

        chart_df = validation_df[
            chart_columns
        ].copy()

        chart_df.columns = [
            "Model",
            "PR-AUC",
        ]

        fig = px.bar(
            chart_df,
            x="Model",
            y="PR-AUC",
            title="Validation PR-AUC by Candidate Model",
            text_auto=".4f",
        )

        fig.update_layout(
            height=450,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

else:

    st.info(
        "Validation comparison artifact is not available."
    )


# ============================================================
# THRESHOLD MONITORING
# ============================================================

st.divider()

st.subheader("⚖️ Decision Threshold Analysis")

st.markdown(
    """
The production decision threshold converts the predicted churn
probability into a binary churn decision.

The current V2 production threshold is **0.20**.
"""
)

if threshold_df is not None and not threshold_df.empty:

    st.dataframe(
        threshold_df,
        use_container_width=True,
        hide_index=True,
    )

    threshold_column = None
    f1_column = None

    for column in threshold_df.columns:

        lower = str(column).lower()

        if (
            "threshold" in lower
            or lower in ["cutoff", "cut_off"]
        ):
            threshold_column = column

        if "f1" in lower:
            f1_column = column

    if (
        threshold_column is not None
        and f1_column is not None
    ):

        plot_df = threshold_df[
            [
                threshold_column,
                f1_column,
            ]
        ].copy()

        plot_df.columns = [
            "Threshold",
            "F1",
        ]

        fig = px.line(
            plot_df,
            x="Threshold",
            y="F1",
            markers=True,
            title="F1 Score Across Decision Thresholds",
        )

        fig.update_layout(
            height=450,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

else:

    st.info(
        "Threshold analysis artifact is not available."
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.divider()

st.subheader("🔎 Feature Importance Monitoring")

if (
    feature_importance_df is not None
    and not feature_importance_df.empty
):

    st.dataframe(
        feature_importance_df,
        use_container_width=True,
        hide_index=True,
    )

    importance_columns = feature_importance_df.columns.tolist()

    feature_col = None
    importance_col = None

    for column in importance_columns:

        lower = str(column).lower()

        if (
            lower in ["feature", "feature_name"]
            or "feature" in lower
        ):
            feature_col = column

        if (
            "importance" in lower
            or "gain" in lower
        ):
            importance_col = column

    if (
        feature_col is not None
        and importance_col is not None
    ):

        plot_df = feature_importance_df[
            [
                feature_col,
                importance_col,
            ]
        ].copy()

        plot_df.columns = [
            "Feature",
            "Importance",
        ]

        plot_df = plot_df.sort_values(
            "Importance",
            ascending=True,
        ).tail(15)

        fig = px.bar(
            plot_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top Model Features",
        )

        fig.update_layout(
            height=550,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

else:

    st.info(
        "Feature importance artifact is not available."
    )


# ============================================================
# MODEL GOVERNANCE
# ============================================================

st.divider()

st.subheader("🛡️ Model Governance")

governance_data = {
    "Component": [
        "Model family",
        "Target",
        "Champion selection",
        "Decision threshold",
        "Feature count",
        "Training pipeline",
        "Explainability",
        "API serving",
    ],
    "Current configuration": [
        "XGBoost",
        "churn_90d",
        "Validation PR-AUC",
        "0.20",
        "19",
        "PySpark + scikit-learn",
        "SHAP TreeExplainer",
        "FastAPI",
    ],
}

governance_df = pd.DataFrame(
    governance_data
)

st.dataframe(
    governance_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# MONITORING STATUS
# ============================================================

st.divider()

st.subheader("🟢 Monitoring Coverage")

monitoring_items = [
    (
        "Model performance",
        "Available",
        "Validation and test metrics are stored as model artifacts.",
    ),
    (
        "Threshold analysis",
        "Available",
        "Threshold performance was evaluated before deployment.",
    ),
    (
        "Feature importance",
        "Available",
        "Model feature importance and SHAP explanations are available.",
    ),
    (
        "Prediction API",
        "Available",
        "FastAPI exposes single and batch prediction endpoints.",
    ),
    (
        "Data drift",
        "Planned",
        "A production reference-vs-current feature monitoring pipeline has not yet been connected.",
    ),
    (
        "Prediction drift",
        "Planned",
        "Production prediction distributions are not yet persisted for continuous monitoring.",
    ),
    (
        "Automated retraining",
        "Planned",
        "Retraining orchestration has not yet been connected to production monitoring.",
    ),
]

monitoring_df = pd.DataFrame(
    monitoring_items,
    columns=[
        "Monitoring Area",
        "Status",
        "Description",
    ],
)

st.dataframe(
    monitoring_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# IMPORTANT LIMITATION
# ============================================================

st.warning(
    """
### Current monitoring limitation

This dashboard monitors the **training and evaluation artifacts** of the
deployed V2 model.

It does not yet claim to perform continuous production drift detection.

For true production monitoring, the next stage would persist live
prediction and feature distributions and calculate metrics such as:

- PSI
- Feature distribution drift
- Prediction probability drift
- Missing-value drift
- Population changes
- Performance degradation once delayed labels become available

This distinction is intentional so that the portfolio does not claim
monitoring capabilities that have not actually been implemented.
"""
)


# ============================================================
# MODEL METADATA
# ============================================================

with st.expander("📋 Raw Model Metadata"):

    if metadata is not None:

        st.json(metadata)

    else:

        st.info(
            "Model metadata is not available."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Enterprise ML Platform • Distributed Churn Intelligence • "
    "Model Monitoring"
)
