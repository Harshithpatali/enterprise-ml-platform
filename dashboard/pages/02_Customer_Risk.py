import requests
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# CONFIG
# ============================================================

API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title="Customer Risk",
    page_icon="🎯",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("🎯 Customer Risk Analysis")

st.markdown(
    """
    Enter a customer's recent behavioral metrics to estimate
    their **90-day churn probability** using the V2 XGBoost model.
    """
)

st.divider()


# ============================================================
# CUSTOMER INPUTS
# ============================================================

st.subheader("Customer Behavior")

col1, col2, col3 = st.columns(3)


with col1:

    order_count = st.number_input(
        "Orders — Last 180 Days",
        min_value=0.0,
        value=1.0,
        step=1.0,
    )

    revenue = st.number_input(
        "Revenue — Last 180 Days",
        min_value=0.0,
        value=100.0,
        step=10.0,
    )

    payment_value = st.number_input(
        "Payment Value — Last 180 Days",
        min_value=0.0,
        value=110.0,
        step=10.0,
    )

    freight_value = st.number_input(
        "Freight Value — Last 180 Days",
        min_value=0.0,
        value=10.0,
        step=1.0,
    )

    item_count = st.number_input(
        "Items — Last 180 Days",
        min_value=0.0,
        value=1.0,
        step=1.0,
    )

    avg_payment_type_count = st.number_input(
        "Avg Payment Type Count",
        min_value=0.0,
        value=1.0,
        step=0.1,
    )

    avg_installments = st.number_input(
        "Avg Installments",
        min_value=0.0,
        value=1.0,
        step=0.1,
    )


with col2:

    avg_review_score = st.number_input(
        "Average Review Score",
        min_value=0.0,
        max_value=5.0,
        value=4.0,
        step=0.1,
    )

    review_count = st.number_input(
        "Review Count",
        min_value=0.0,
        value=1.0,
        step=1.0,
    )

    avg_delivery_delay = st.number_input(
        "Average Delivery Delay (Days)",
        value=0.0,
        step=1.0,
    )

    late_delivery_rate = st.number_input(
        "Late Delivery Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
    )

    unique_products = st.number_input(
        "Unique Products",
        min_value=0.0,
        value=1.0,
        step=1.0,
    )

    unique_categories = st.number_input(
        "Unique Categories",
        min_value=0.0,
        value=1.0,
        step=1.0,
    )

    unique_sellers = st.number_input(
        "Unique Sellers",
        min_value=0.0,
        value=1.0,
        step=1.0,
    )


with col3:

    recency_days = st.number_input(
        "Recency (Days)",
        min_value=0.0,
        value=20.0,
        step=1.0,
    )

    customer_lifetime_days = st.number_input(
        "Customer Lifetime (Days)",
        min_value=0.0,
        value=20.0,
        step=1.0,
    )

    avg_order_value = st.number_input(
        "Average Order Value",
        min_value=0.0,
        value=100.0,
        step=10.0,
    )

    orders_per_month = st.number_input(
        "Orders Per Month",
        min_value=0.0,
        value=0.2,
        step=0.1,
    )

    revenue_per_month = st.number_input(
        "Revenue Per Month",
        min_value=0.0,
        value=20.0,
        step=5.0,
    )


# ============================================================
# PREDICTION
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Customer Risk",
    type="primary",
    use_container_width=True,
)


if predict_button:

    payload = {
        "order_count_180d": order_count,
        "revenue_180d": revenue,
        "payment_value_180d": payment_value,
        "freight_value_180d": freight_value,
        "item_count_180d": item_count,
        "avg_payment_type_count_180d": avg_payment_type_count,
        "avg_installments_180d": avg_installments,
        "avg_review_score_180d": avg_review_score,
        "review_count_180d": review_count,
        "avg_delivery_delay_180d": avg_delivery_delay,
        "late_delivery_rate_180d": late_delivery_rate,
        "unique_products_180d": unique_products,
        "unique_categories_180d": unique_categories,
        "unique_sellers_180d": unique_sellers,
        "recency_days": recency_days,
        "customer_lifetime_days": customer_lifetime_days,
        "avg_order_value_180d": avg_order_value,
        "orders_per_month_180d": orders_per_month,
        "revenue_per_month_180d": revenue_per_month,
    }

    try:

        with st.spinner("Calling churn prediction API..."):

            response = requests.post(
                API_URL,
                json=payload,
                timeout=30,
            )

        if response.status_code != 200:

            st.error(
                f"Prediction API returned "
                f"HTTP {response.status_code}"
            )

            st.code(response.text)

        else:

            result = response.json()

            probability = result[
                "churn_probability"
            ]

            prediction = result[
                "churn_prediction"
            ]

            risk_band = result[
                "risk_band"
            ]

            threshold = result[
                "threshold"
            ]


            # ====================================================
            # RESULT HEADER
            # ====================================================

            st.divider()

            st.subheader(
                "Prediction Result"
            )

            metric1, metric2, metric3 = st.columns(3)

            with metric1:

                st.metric(
                    "Churn Probability",
                    f"{probability:.2%}",
                )

            with metric2:

                st.metric(
                    "Prediction",
                    "Likely Churn"
                    if prediction == 1
                    else "Likely Retained",
                )

            with metric3:

                st.metric(
                    "Risk Band",
                    risk_band,
                )


            # ====================================================
            # GAUGE
            # ====================================================

            gauge_col, insight_col = st.columns(
                [1, 1]
            )

            with gauge_col:

                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=probability * 100,
                        number={
                            "suffix": "%"
                        },
                        title={
                            "text": "90-Day Churn Risk"
                        },
                        gauge={
                            "axis": {
                                "range": [0, 100]
                            },
                            "threshold": {
                                "line": {
                                    "width": 4
                                },
                                "value": threshold * 100,
                            },
                            "steps": [
                                {
                                    "range": [0, 40],
                                },
                                {
                                    "range": [40, 70],
                                },
                                {
                                    "range": [70, 100],
                                },
                            ],
                        },
                    )
                )

                fig.update_layout(
                    height=350,
                    margin=dict(
                        l=20,
                        r=20,
                        t=70,
                        b=20,
                    ),
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )


            # ====================================================
            # BUSINESS INTERPRETATION
            # ====================================================

            with insight_col:

                st.subheader(
                    "Business Interpretation"
                )

                if risk_band == "High":

                    st.error(
                        "High-risk customer"
                    )

                    st.markdown(
                        """
                        **Recommended action**

                        - Prioritize retention outreach
                        - Review recent customer experience
                        - Consider personalized offers
                        - Monitor this customer closely
                        """
                    )

                elif risk_band == "Medium":

                    st.warning(
                        "Medium-risk customer"
                    )

                    st.markdown(
                        """
                        **Recommended action**

                        - Add to retention monitoring
                        - Consider targeted engagement
                        - Monitor purchasing frequency
                        - Review customer satisfaction
                        """
                    )

                else:

                    st.success(
                        "Low-risk customer"
                    )

                    st.markdown(
                        """
                        **Recommended action**

                        - Maintain engagement
                        - Encourage repeat purchases
                        - Continue personalized recommendations
                        - Monitor behavior over time
                        """ 
                    )


            # ====================================================
            # CUSTOMER PROFILE
            # ====================================================

            st.divider()

            st.subheader(
                "Customer Profile"
            )

            profile_col1, profile_col2, profile_col3, profile_col4 = (
                st.columns(4)
            )

            with profile_col1:

                st.metric(
                    "Orders",
                    f"{order_count:.0f}",
                )

            with profile_col2:

                st.metric(
                    "Revenue",
                    f"₹{revenue:,.2f}",
                )

            with profile_col3:

                st.metric(
                    "Recency",
                    f"{recency_days:.0f} days",
                )

            with profile_col4:

                st.metric(
                    "Avg Order Value",
                    f"₹{avg_order_value:,.2f}",
                )


            # ====================================================
            # API STATUS
            # ====================================================

            st.caption(
                f"Prediction generated by FastAPI → "
                f"XGBoost V2 | Threshold: {threshold:.2f}"
            )


    except requests.exceptions.ConnectionError:

        st.error(
            """
            ❌ Could not connect to the FastAPI backend.

            Make sure the API is running:

            `uvicorn api.main:app --reload --host 0.0.0.0 --port 8000`
            """
        )

    except requests.exceptions.Timeout:

        st.error(
            "The prediction API timed out."
        )

    except Exception as exc:

        st.error(
            f"Unexpected error: {exc}"
        )