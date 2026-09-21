import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

st.set_page_config(
    page_title="Enterprise Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LOAD MODEL METADATA
# ============================================================

@st.cache_data
def load_metadata():

    metadata_path = MODEL_DIR / "model_metadata.json"

    with open(metadata_path, "r") as f:
        return json.load(f)


@st.cache_data
def load_validation_results():

    path = MODEL_DIR / "validation_results.csv"

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


@st.cache_data
def load_feature_importance():

    path = MODEL_DIR / "feature_importance.csv"

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


metadata = load_metadata()
validation_results = load_validation_results()
feature_importance = load_feature_importance()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("Enterprise ML")

    st.markdown(
        """
        ### Churn Intelligence

        **Model:** XGBoost  
        **Target:** 90-Day Churn  
        **Pipeline:** PySpark → XGBoost → FastAPI
        """
    )

    st.divider()

    st.success("Model API Ready")

    st.caption(
        "V2 production model"
    )


# ============================================================
# HEADER
# ============================================================

st.title("📊 Enterprise Churn Intelligence")

st.markdown(
    """
    ### Customer Churn Prediction & Retention Intelligence

    Production-style machine learning platform built using
    customer transaction behavior from the Olist e-commerce dataset.
    """
)


# ============================================================
# KPI SECTION
# ============================================================

test_metrics = metadata["test_metrics"]

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Model",
        metadata["champion_model"].upper(),
    )

with col2:
    st.metric(
        "ROC-AUC",
        f"{test_metrics['roc_auc']:.3f}",
    )

with col3:
    st.metric(
        "PR-AUC",
        f"{test_metrics['pr_auc']:.3f}",
    )

with col4:
    st.metric(
        "F1 Score",
        f"{test_metrics['f1']:.3f}",
    )

with col5:
    st.metric(
        "Threshold",
        f"{metadata['threshold']:.2f}",
    )


st.divider()


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader("Model Performance")

left, right = st.columns(2)


with left:

    if not validation_results.empty:

        st.markdown("#### Validation Model Comparison")

        metric_options = [
            "pr_auc",
            "roc_auc",
            "f1",
            "precision",
            "recall",
        ]

        available_metrics = [
            m
            for m in metric_options
            if m in validation_results.columns
        ]

        selected_metric = st.selectbox(
            "Metric",
            available_metrics,
        )

        chart_data = validation_results.copy()

        fig = px.bar(
            chart_data,
            x="model",
            y=selected_metric,
            text_auto=".3f",
            title=f"Validation {selected_metric.upper()}",
        )

        fig.update_layout(
            height=420,
            xaxis_title="Model",
            yaxis_title=selected_metric.upper(),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    else:

        st.info(
            "Validation results are not available."
        )


with right:

    st.markdown("#### Champion Test Metrics")

    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "ROC-AUC",
                "PR-AUC",
                "Precision",
                "Recall",
                "F1",
            ],
            "Score": [
                test_metrics["roc_auc"],
                test_metrics["pr_auc"],
                test_metrics["precision"],
                test_metrics["recall"],
                test_metrics["f1"],
            ],
        }
    )

    fig = px.bar(
        metrics_df,
        x="Metric",
        y="Score",
        text_auto=".3f",
        title="XGBoost Test Performance",
    )

    fig.update_yaxes(
        range=[0, 1]
    )

    fig.update_layout(
        height=420,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.divider()

st.subheader(
    "🔎 Model Feature Importance"
)

if not feature_importance.empty:

    st.caption(
        "Features used by the V2 XGBoost churn model."
    )

    importance_columns = feature_importance.columns.tolist()

    # Detect the feature and importance columns
    feature_column = next(
        (
            c
            for c in importance_columns
            if c.lower()
            in ["feature", "feature_name"]
        ),
        importance_columns[0],
    )

    importance_column = next(
        (
            c
            for c in importance_columns
            if "importance" in c.lower()
        ),
        importance_columns[-1],
    )

    plot_df = (
        feature_importance[
            [feature_column, importance_column]
        ]
        .sort_values(
            importance_column,
            ascending=False,
        )
        .head(15)
        .sort_values(
            importance_column,
            ascending=True,
        )
    )

    fig = px.bar(
        plot_df,
        x=importance_column,
        y=feature_column,
        orientation="h",
        title="Top 15 Features",
        text_auto=".3f",
    )

    fig.update_layout(
        height=600,
        yaxis_title="Feature",
        xaxis_title="Importance",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

else:

    st.info(
        "Feature importance artifact not found."
    )


# ============================================================
# ARCHITECTURE
# ============================================================

st.divider()

st.subheader("🏗️ ML Platform Architecture")

architecture = """
Raw Olist Data
        ↓
Hadoop HDFS
        ↓
PySpark Bronze Layer
        ↓
PySpark Silver Layer
        ↓
Customer Feature Engineering
        ↓
V2 ML Dataset
        ↓
XGBoost Champion
        ↓
SHAP Explainability
        ↓
FastAPI Inference
        ↓
Streamlit Intelligence Dashboard
"""

st.code(
    architecture,
    language="text",
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Enterprise ML Platform • V2 Churn Intelligence • "
    "PySpark + Hadoop + XGBoost + SHAP + FastAPI + Streamlit"
)
