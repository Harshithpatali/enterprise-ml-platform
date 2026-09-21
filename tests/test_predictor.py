import pytest

from src.predictor import (
    feature_names,
    get_risk_band,
    predict_customer,
)


def make_sample_customer():
    return {
        "order_count_180d": 2,
        "revenue_180d": 250.0,
        "payment_value_180d": 270.0,
        "freight_value_180d": 35.0,
        "item_count_180d": 3,
        "avg_payment_type_count_180d": 1.0,
        "avg_installments_180d": 2.0,
        "avg_review_score_180d": 4.0,
        "review_count_180d": 2,
        "avg_delivery_delay_180d": 1.0,
        "late_delivery_rate_180d": 0.1,
        "unique_products_180d": 2,
        "unique_categories_180d": 2,
        "unique_sellers_180d": 2,
        "recency_days": 30,
        "customer_lifetime_days": 60,
        "avg_order_value_180d": 125.0,
        "orders_per_month_180d": 0.67,
        "revenue_per_month_180d": 83.33,
    }


def test_feature_schema_contains_expected_features():
    assert len(feature_names) == 19

    assert "order_count_180d" in feature_names
    assert "recency_days" in feature_names
    assert "revenue_180d" in feature_names
    assert "avg_order_value_180d" in feature_names


def test_risk_band_low():
    assert get_risk_band(0.10) == "Low"


def test_risk_band_medium():
    assert get_risk_band(0.50) == "Medium"


def test_risk_band_high():
    assert get_risk_band(0.80) == "High"


def test_risk_band_boundaries():
    assert get_risk_band(0.39) == "Low"
    assert get_risk_band(0.40) == "Medium"
    assert get_risk_band(0.69) == "Medium"
    assert get_risk_band(0.70) == "High"


def test_prediction_returns_expected_schema():

    customer = make_sample_customer()

    result = predict_customer(customer)

    assert "churn_probability" in result
    assert "churn_prediction" in result
    assert "risk_band" in result
    assert "threshold" in result


def test_prediction_probability_is_valid():

    customer = make_sample_customer()

    result = predict_customer(customer)

    probability = result["churn_probability"]

    assert 0.0 <= probability <= 1.0


def test_prediction_is_binary():

    customer = make_sample_customer()

    result = predict_customer(customer)

    assert result["churn_prediction"] in [0, 1]


def test_prediction_threshold_matches_metadata():

    customer = make_sample_customer()

    result = predict_customer(customer)

    probability = result["churn_probability"]
    prediction = result["churn_prediction"]
    threshold = result["threshold"]

    expected_prediction = int(
        probability >= threshold
    )

    assert prediction == expected_prediction


def test_missing_feature_is_rejected():

    customer = make_sample_customer()

    customer.pop("order_count_180d")

    with pytest.raises(ValueError):

        predict_customer(customer)
