from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="SHAP Explainability",
    page_icon="🔍",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SHAP_DIR = PROJECT_ROOT / "models" / "shap"

IMPORTANCE_PATH = SHAP_DIR / "shap_feature_importance.csv"
SUMMARY_IMAGE_PATH = SHAP_DIR / "shap_summary.png"
BAR_IMAGE_PATH = SHAP_DIR / "shap_bar.png"
SHAP_VALUES_PATH = SHAP_DIR / "shap_values_sample.csv"
HIGH_RISK_PATH = SHAP_DIR / "high_risk_customers_sample.csv"


# ============================================================
# HEADER
# ============================================================

st.title("🔍 SHAP Explainability")
st.caption(
    "Model interpretability for the Enterprise Churn Intelligence platform"
)

st.markdown(
    """
SHAP (SHapley Additive exPlanations) helps explain **which features are
driving the churn model's predictions**.

The analysis below is based on the SHAP artifacts generated from the
deployed V2 XGBoost champion model.
"""
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_importance():
    if not IMPORTANCE_PATH.exists():
        return None

    return pd.read_csv(IMPORTANCE_PATH)


@st.cache_data
def load_shap_values():
    if not SHAP_VALUES_PATH.exists():
        return None

    return pd.read_csv(SHAP_VALUES_PATH)


@st.cache_data
def load_high_risk():
    if not HIGH_RISK_PATH.exists():
        return None

    return pd.read_csv(HIGH_RISK_PATH)


importance_df = load_importance()
shap_values_df = load_shap_values()
high_risk_df = load_high_risk()


if importance_df is None:
    st.error(
        f"SHAP feature importance file was not found:\n\n"
        f"`{IMPORTANCE_PATH}`"
    )
    st.stop()


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

importance_df.columns = [
    str(column).strip() for column in importance_df.columns
]

# Expected columns from the training pipeline:
# feature, mean_abs_shap, importance_pct

feature_column = None
shap_column = None
percentage_column = None

for column in importance_df.columns:
    lower = column.lower()

    if lower in ["feature", "feature_name"]:
        feature_column = column

    if "mean_abs" in lower and "shap" in lower:
        shap_column = column

    if "importance_pct" in lower or "percentage" in lower:
        percentage_column = column


if feature_column is None:
    feature_column = importance_df.columns[0]

if shap_column is None:
    numeric_columns = importance_df.select_dtypes(
        include="number"
    ).columns.tolist()

    if numeric_columns:
        shap_column = numeric_columns[0]


if shap_column is None:
    st.error("Could not identify the SHAP importance column.")
    st.stop()


importance_df = importance_df.sort_values(
    shap_column,
    ascending=False
).reset_index(drop=True)


# ============================================================
# KPI SECTION
# ============================================================

total_features = len(importance_df)

top_feature = importance_df.iloc[0][feature_column]
top_value = float(importance_df.iloc[0][shap_column])

if percentage_column is not None:
    top_percentage = float(
        importance_df.iloc[0][percentage_column]
    )
else:
    total_shap = importance_df[shap_column].sum()
    top_percentage = (
        (top_value / total_shap) * 100
        if total_shap > 0
        else 0
    )


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Features Explained",
        total_features
    )

with col2:
    st.metric(
        "Top Driver",
        top_feature
    )

with col3:
    st.metric(
        "Top Mean |SHAP|",
        f"{top_value:.4f}"
    )

with col4:
    st.metric(
        "Top Driver Contribution",
        f"{top_percentage:.1f}%"
    )


st.divider()


# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================

st.subheader("📊 Global Feature Importance")

st.markdown(
    """
Mean absolute SHAP value measures the average magnitude of a feature's
contribution to model predictions.

A larger value means the feature has a greater influence on the model's
predictions across the analyzed customer sample.
"""
)

display_df = importance_df.copy()

display_df["Feature"] = display_df[feature_column]

display_df["Mean |SHAP|"] = display_df[shap_column]

if percentage_column is not None:
    display_df["Contribution %"] = display_df[
        percentage_column
    ]

display_columns = [
    "Feature",
    "Mean |SHAP|",
]

if "Contribution %" in display_df.columns:
    display_columns.append("Contribution %")

st.dataframe(
    display_df[display_columns],
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# INTERACTIVE FEATURE IMPORTANCE
# ============================================================

st.subheader("📈 Interactive SHAP Importance")

top_n = st.slider(
    "Number of features",
    min_value=5,
    max_value=min(19, total_features),
    value=min(15, total_features),
    step=1,
)


chart_df = importance_df.head(top_n).copy()

chart_df["Feature"] = chart_df[feature_column]
chart_df["Mean |SHAP|"] = chart_df[shap_column]

chart_df = chart_df.sort_values(
    "Mean |SHAP|",
    ascending=True
)

fig = px.bar(
    chart_df,
    x="Mean |SHAP|",
    y="Feature",
    orientation="h",
    title=f"Top {top_n} Churn Drivers",
    labels={
        "Mean |SHAP|": "Mean Absolute SHAP Value",
        "Feature": "Feature",
    },
)

fig.update_layout(
    height=max(500, top_n * 35),
    margin=dict(l=20, r=20, t=60, b=20),
)

st.plotly_chart(
    fig,
    use_container_width=True,
)


# ============================================================
# SAVED SHAP VISUALIZATIONS
# ============================================================

st.subheader("🧠 SHAP Model Visualizations")

col1, col2 = st.columns(2)

with col1:

    if BAR_IMAGE_PATH.exists():

        st.markdown("**SHAP Feature Importance**")

        st.image(
            str(BAR_IMAGE_PATH),
            use_container_width=True,
        )

    else:

        st.info(
            "Saved SHAP bar visualization is not available."
        )


with col2:

    if SUMMARY_IMAGE_PATH.exists():

        st.markdown("**SHAP Summary Plot**")

        st.image(
            str(SUMMARY_IMAGE_PATH),
            use_container_width=True,
        )

    else:

        st.info(
            "Saved SHAP summary visualization is not available."
        )


# ============================================================
# TOP DRIVERS
# ============================================================

st.divider()

st.subheader("🎯 Top Churn Drivers")

top_features = importance_df.head(5)

for index, row in top_features.iterrows():

    feature = row[feature_column]
    shap_value = float(row[shap_column])

    if percentage_column is not None:
        contribution = float(row[percentage_column])
        contribution_text = f"{contribution:.2f}%"
    else:
        contribution_text = "N/A"

    st.markdown(
        f"""
**{index + 1}. `{feature}`**

- Mean absolute SHAP: `{shap_value:.4f}`
- Relative contribution: `{contribution_text}`
"""
    )


# ============================================================
# BUSINESS INTERPRETATION
# ============================================================

st.divider()

st.subheader("💼 Business Interpretation")

st.markdown(
    """
### How to interpret these results

The SHAP analysis identifies the variables that the trained model relies
on most heavily when distinguishing customers according to predicted
90-day churn risk.

For the current V2 model, the strongest global drivers include:

- **Order frequency / order count**
- **Payment value**
- **Item count**
- **Recency**
- **Review behavior**
- **Freight value**
- **Average order value**
- **Customer lifetime**

These relationships describe **model behavior**, not causal effects.

For example, if order frequency has a high SHAP importance, this means
the model uses order frequency strongly when generating predictions. It
does not by itself establish that changing order frequency would cause
the customer's churn probability to change.
"""
)


# ============================================================
# HIGH-RISK CUSTOMER SAMPLE
# ============================================================

st.divider()

st.subheader("🚨 High-Risk Customer Sample")

if high_risk_df is not None and not high_risk_df.empty:

    st.markdown(
        """
The following customers were included in the SHAP analysis sample and
were identified as high-risk by the model.
"""
    )

    st.dataframe(
        high_risk_df,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "No high-risk customer SHAP sample is available."
    )


# ============================================================
# SHAP SAMPLE DATA
# ============================================================

with st.expander("🔬 View SHAP Sample Data"):

    if shap_values_df is not None:

        st.write(
            f"Rows available: **{len(shap_values_df):,}**"
        )

        st.dataframe(
            shap_values_df.head(100),
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "SHAP sample values are not available."
        )


# ============================================================
# METHODOLOGY
# ============================================================

with st.expander("📚 Methodology"):

    st.markdown(
        """
### SHAP methodology

The deployed model is an **XGBoost classifier inside a scikit-learn
Pipeline**.

The explainability workflow:

1. Load the production champion model.
2. Extract the fitted preprocessing pipeline.
3. Apply the same preprocessing used during training.
4. Extract the XGBoost estimator.
5. Use `TreeExplainer` to calculate SHAP values.
6. Calculate mean absolute SHAP values.
7. Rank features by their global contribution.
8. Generate summary and bar visualizations.

The SHAP analysis was calculated on a sample of **5,000 test
observations**.

This provides a model-level explanation of the deployed V2 classifier.
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Enterprise ML Platform • Distributed Churn Intelligence • "
    "V2 XGBoost Champion Model"
)
