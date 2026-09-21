import pandas as pd

from .model_loader import (
    model,
    feature_names,
    THRESHOLD,
)


def validate_features(data: dict) -> None:
    """
    Validate that the incoming prediction request
    contains exactly the features expected by V2.
    """

    missing = [
        feature
        for feature in feature_names
        if feature not in data
    ]

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )


def predict_customer(data: dict) -> dict:
    """
    Generate a churn prediction for one customer.
    """

    validate_features(data)

    # Preserve the exact feature ordering used during training.
    X = pd.DataFrame(
        [[data[feature] for feature in feature_names]],
        columns=feature_names,
    )

    # Probability of churn
    churn_probability = float(
        model.predict_proba(X)[0][1]
    )

    # Business threshold from V2 model metadata
    churn_prediction = int(
        churn_probability >= THRESHOLD
    )

    risk_band = get_risk_band(churn_probability)

    return {
        "churn_probability": round(
            churn_probability, 6
        ),
        "churn_prediction": churn_prediction,
        "risk_band": risk_band,
        "threshold": THRESHOLD,
    }


def get_risk_band(probability: float) -> str:
    """
    Convert churn probability into a business risk band.
    """

    if probability < 0.40:
        return "Low"

    if probability < 0.70:
        return "Medium"

    return "High"
