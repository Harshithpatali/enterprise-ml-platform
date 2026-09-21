from io import BytesIO, StringIO

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File

from api.schemas import (
    CustomerFeatures,
    PredictionResponse,
    HealthResponse,
)
from io import StringIO, BytesIO

import pandas as pd

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
)

from fastapi.responses import StreamingResponse
from src.model_loader import (
    MODEL_NAME,
    TARGET,
    THRESHOLD,
    feature_names,
)

from src.predictor import predict_customer


app = FastAPI(
    title="Enterprise Churn Intelligence API",
    description=(
        "Production-style customer churn prediction API "
        "powered by the V2 XGBoost model."
    ),
    version="1.0.0",
)


@app.get(
    "/",
    tags=["System"],
)
def root():
    return {
        "service": "Enterprise Churn Intelligence API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
def health():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "target": TARGET,
        "threshold": THRESHOLD,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
)
def predict(customer: CustomerFeatures):

    try:
        result = predict_customer(
            customer.model_dump()
        )

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@app.post(
    "/predict/batch",
    tags=["Batch Prediction"],
)
async def predict_batch(
    file: UploadFile = File(...)
):
    """
    Upload a CSV containing the 19 V2 model features
    and receive a downloadable CSV with predictions.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file provided.",
        )

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported.",
        )

    try:
        contents = await file.read()

        df = pd.read_csv(
            StringIO(contents.decode("utf-8"))
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to read CSV: {exc}",
        )

    missing_columns = [
        feature
        for feature in feature_names
        if feature not in df.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "CSV is missing required model features.",
                "missing_columns": missing_columns,
            },
        )

    try:
        X = df[feature_names].copy()

        probabilities = model_predict_proba(X)

        df["churn_probability"] = probabilities
        df["churn_prediction"] = (
            df["churn_probability"] >= THRESHOLD
        ).astype(int)

        df["risk_band"] = df[
            "churn_probability"
        ].apply(get_risk_band)

        # Create CSV in memory
        output = BytesIO()

        df.to_csv(
            output,
            index=False,
        )

        output.seek(0)

        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={
                "Content-Disposition": (
                    'attachment; filename="churn_predictions.csv"'
                )
            },
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {exc}",
        )


def predict_batch_dataframe(
    X: pd.DataFrame,
):
    """
    Run batch inference using the existing
    V2 model pipeline.
    """

    probabilities = model_predict_proba(X)

    return probabilities


def model_predict_proba(
    X: pd.DataFrame,
):
    """
    Wrapper around the loaded V2 model.
    """

    from src.model_loader import model

    return model.predict_proba(X)[:, 1]


def get_risk_band(
    probability: float,
) -> str:

    if probability < 0.40:
        return "Low"

    if probability < 0.70:
        return "Medium"

    return "High"