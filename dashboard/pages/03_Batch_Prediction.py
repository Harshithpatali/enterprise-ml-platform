import io
import os

import pandas as pd
import requests
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")
BATCH_PREDICT_URL = f"{API_BASE_URL.rstrip('/')}/predict/batch"

st.set_page_config(
    page_title="Batch Prediction",
    page_icon="📦",
    layout="wide",
)


# ============================================================
# REQUIRED V2 FEATURES
# ============================================================

FEATURES = [
    "order_count_180d",
    "revenue_180d",
    "payment_value_180d",
    "freight_value_180d",
    "item_count_180d",
    "avg_payment_type_count_180d",
    "avg_installments_180d",
    "avg_review_score_180d",
    "review_count_180d",
    "avg_delivery_delay_180d",
    "late_delivery_rate_180d",
    "unique_products_180d",
    "unique_categories_180d",
    "unique_sellers_180d",
    "recency_days",
    "customer_lifetime_days",
    "avg_order_value_180d",
    "orders_per_month_180d",
    "revenue_per_month_180d",
]


# ============================================================
# HEADER
# ============================================================

st.title("📦 Batch Customer Prediction")

st.markdown(
    """
    Upload a customer CSV and run the **V2 XGBoost churn model**
    across multiple customers through the FastAPI inference service.
    """
)

st.divider()


# ============================================================
# FILE UPLOAD
# ============================================================

st.subheader("Upload Customer Data")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="CSV must contain all 19 V2 model features.",
)


# ============================================================
# PROCESS UPLOAD
# ============================================================

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

    except Exception as exc:
        st.error(f"Unable to read CSV: {exc}")
        st.stop()

    st.success(
        f"File loaded successfully: "
        f"{len(df):,} rows × {len(df.columns):,} columns"
    )

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in FEATURES
        if column not in df.columns
    ]

    if missing_columns:

        st.error(
            "The uploaded CSV is missing required model features."
        )

        st.write("Missing columns:")

        st.code(
            "\n".join(missing_columns)
        )

        st.stop()

    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------

    st.subheader("Data Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Dataset summary
    # --------------------------------------------------------

    st.subheader("Dataset Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Customers",
            f"{len(df):,}",
        )

    with col2:
        st.metric(
            "Features",
            len(FEATURES),
        )

    with col3:
        missing_rows = (
            df[FEATURES]
            .isna()
            .any(axis=1)
            .sum()
        )

        st.metric(
            "Rows with Missing Values",
            f"{missing_rows:,}",
        )

    # --------------------------------------------------------
    # Run prediction
    # --------------------------------------------------------

    st.divider()

    predict_button = st.button(
        "🚀 Run Batch Prediction",
        type="primary",
        use_container_width=True,
    )

    if predict_button:

        uploaded_file.seek(0)

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "text/csv",
            )
        }

        try:

            with st.spinner(
                "Sending data to FastAPI..."
            ):

                response = requests.post(
                    API_URL,
                    files=files,
                    timeout=120,
                )

            if response.status_code != 200:

                st.error(
                    f"API returned HTTP "
                    f"{response.status_code}"
                )

                try:
                    st.json(response.json())
                except Exception:
                    st.code(response.text)

            else:

                result_df = pd.read_csv(
                    io.BytesIO(response.content)
                )

                st.session_state[
                    "batch_results"
                ] = result_df

                st.success(
                    f"Prediction completed for "
                    f"{len(result_df):,} customers."
                )

        except requests.exceptions.ConnectionError:

            st.error(
                """
                ❌ Could not connect to FastAPI.

                Make sure the backend is running:

                `uvicorn api.main:app --reload --host 0.0.0.0 --port 8000`
                """
            )

        except requests.exceptions.Timeout:

            st.error(
                "The prediction API timed out."
            )

        except Exception as exc:

            st.error(
                f"Batch prediction failed: {exc}"
            )


# ============================================================
# RESULTS
# ============================================================

if "batch_results" in st.session_state:

    result_df = st.session_state["batch_results"]

    st.divider()

    st.subheader("📊 Prediction Results")

    # --------------------------------------------------------
    # Summary metrics
    # --------------------------------------------------------

    total = len(result_df)

    high_risk = (
        result_df["risk_band"]
        .eq("High")
        .sum()
    )

    medium_risk = (
        result_df["risk_band"]
        .eq("Medium")
        .sum()
    )

    low_risk = (
        result_df["risk_band"]
        .eq("Low")
        .sum()
    )

    predicted_churn = (
        result_df["churn_prediction"]
        .eq(1)
        .sum()
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Customers",
            f"{total:,}",
        )

    with col2:
        st.metric(
            "Predicted Churn",
            f"{predicted_churn:,}",
        )

    with col3:
        st.metric(
            "High Risk",
            f"{high_risk:,}",
        )

    with col4:
        st.metric(
            "Medium Risk",
            f"{medium_risk:,}",
        )

    # --------------------------------------------------------
    # Risk distribution
    # --------------------------------------------------------

    st.subheader("Risk Distribution")

    risk_df = pd.DataFrame(
        {
            "Risk Band": [
                "High",
                "Medium",
                "Low",
            ],
            "Customers": [
                high_risk,
                medium_risk,
                low_risk,
            ],
        }
    )

    st.bar_chart(
        risk_df.set_index("Risk Band")
    )

    # --------------------------------------------------------
    # Prediction table
    # --------------------------------------------------------

    st.subheader("Customer Predictions")

    display_columns = [
        "churn_probability",
        "churn_prediction",
        "risk_band",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in result_df.columns
    ]

    st.dataframe(
        result_df[available_columns],
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    csv_data = result_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Predictions CSV",
        data=csv_data,
        file_name="churn_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )