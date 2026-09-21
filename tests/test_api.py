from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


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


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert "service" in data
    assert "status" in data


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == "xgboost"
    assert data["target"] == "churn_90d"


def test_single_prediction():
    customer = make_sample_customer()

    response = client.post(
        "/predict",
        json=customer,
    )

    assert response.status_code == 200

    data = response.json()

    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "risk_band" in data
    assert "threshold" in data

    assert 0.0 <= data["churn_probability"] <= 1.0

    assert data["churn_prediction"] in [0, 1]

    assert data["risk_band"] in [
        "Low",
        "Medium",
        "High",
    ]


def test_prediction_rejects_missing_features():
    customer = make_sample_customer()

    customer.pop("order_count_180d")

    response = client.post(
        "/predict",
        json=customer,
    )

    assert response.status_code == 422


def test_prediction_rejects_invalid_feature_type():
    customer = make_sample_customer()

    customer["order_count_180d"] = "invalid"

    response = client.post(
        "/predict",
        json=customer,
    )

    assert response.status_code == 422


def test_batch_endpoint_requires_file():
    response = client.post("/predict/batch")

    assert response.status_code == 422
