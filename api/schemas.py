from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):

    order_count_180d: float = Field(..., ge=0)
    revenue_180d: float = Field(..., ge=0)
    payment_value_180d: float = Field(..., ge=0)
    freight_value_180d: float = Field(..., ge=0)
    item_count_180d: float = Field(..., ge=0)

    avg_payment_type_count_180d: float = Field(..., ge=0)
    avg_installments_180d: float = Field(..., ge=0)
    avg_review_score_180d: float = Field(..., ge=0)
    review_count_180d: float = Field(..., ge=0)

    avg_delivery_delay_180d: float
    late_delivery_rate_180d: float = Field(..., ge=0, le=1)

    unique_products_180d: float = Field(..., ge=0)
    unique_categories_180d: float = Field(..., ge=0)
    unique_sellers_180d: float = Field(..., ge=0)

    recency_days: float = Field(..., ge=0)
    customer_lifetime_days: float = Field(..., ge=0)

    avg_order_value_180d: float = Field(..., ge=0)
    orders_per_month_180d: float = Field(..., ge=0)
    revenue_per_month_180d: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_band: str
    threshold: float


class HealthResponse(BaseModel):
    status: str
    model: str
    target: str
    threshold: float
