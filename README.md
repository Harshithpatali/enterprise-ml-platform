# Enterprise ML Platform — Distributed Churn Intelligence & Model Monitoring

Production-style end-to-end ML platform combining Hadoop HDFS, PySpark, temporal churn modeling, XGBoost, SHAP, FastAPI, Streamlit, and automated tests.

Repository: https://github.com/Harshithpatali/enterprise-ml-platform

## Overview

Raw Olist e-commerce data flows through Hadoop HDFS and PySpark Bronze/Silver/Gold layers into a temporal customer-level ML dataset, then into an XGBoost prediction service and Streamlit MLOps dashboard.

Architecture:

~~~text
Olist -> HDFS -> PySpark Bronze/Silver/Gold -> ML Dataset -> XGBoost -> SHAP
                                                       |
                                                       v
                                                FastAPI -> Streamlit
~~~

## Business Problem

Predict whether a customer will make no completed purchase during the next 90 days.

- Observation window: previous 180 days
- Prediction horizon: next 90 days
- Analytical customer key: customer_unique_id
- Target: churn_90d
- Model selection metric: validation PR-AUC
- Decision threshold: 0.20, selected on validation F1

## Data Engineering

The pipeline uses Hadoop HDFS and PySpark with Raw, Bronze, Silver, Gold, and ML Dataset layers. It processes the Olist customers, orders, order items, payments, reviews, products, sellers, geolocation, and category translation datasets.

## Model Development

V2 compares Logistic Regression, Random Forest, and XGBoost.

| Model | Validation PR-AUC | Validation ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.989005 | 0.881206 |
| Random Forest | 0.992064 | 0.904257 |
| **XGBoost** | **0.992599** | **0.906689** |

XGBoost was selected using validation PR-AUC.

### Locked test performance

At threshold 0.20:

| Metric | Test |
|---|---:|
| PR-AUC | 0.991381 |
| ROC-AUC | 0.879110 |
| Precision | 0.988654 |
| Recall | 1.000000 |
| F1 | 0.994295 |

Confusion matrix:

~~~text
                 Predicted
                 0       1
Actual 0       520     225
Actual 1         0   19606
~~~

## Modeling Audit

The V2 structural audit found a highly imbalanced population dominated by customers with one recent purchase.

- 95,194 customer-snapshot rows
- 93,358 unique customers
- 91,452 churn observations
- 3,742 retained observations
- Overall churn rate: approximately 96.1%
- Most customer histories contain one order
- order_count_180d is the strongest model feature

Therefore, the high PR-AUC/F1 should not be interpreted as proof that the model perfectly predicts churn among an already-active repeat-customer population. The model is temporally constructed and leakage-audited, but population structure limits how broadly the metrics should be interpreted. A future iteration should evaluate a more explicitly active-customer cohort.

## Feature Engineering

The deployed V2 model uses 19 temporal customer features including order_count_180d, revenue_180d, payment_value_180d, item_count_180d, review metrics, delivery metrics, unique products/categories/sellers, recency_days, customer_lifetime_days, average order value, orders per month, and revenue per month.

## SHAP Explainability

SHAP TreeExplainer is applied to the trained XGBoost model.

| Feature | Mean Absolute SHAP | Share |
|---|---:|---:|
| order_count_180d | 0.782484 | 42.38% |
| payment_value_180d | 0.332385 | 18.00% |
| item_count_180d | 0.217530 | 11.78% |
| recency_days | 0.087093 | 4.72% |
| avg_review_score_180d | 0.071867 | 3.89% |

The dashboard includes global importance, saved SHAP visualizations, top drivers, and high-risk customer examples.

## FastAPI

Endpoints:

- GET /health — service and model status
- POST /predict — single-customer prediction
- POST /predict/batch — CSV batch scoring with downloadable results
- GET /docs — Swagger UI

Predictions include churn probability, binary prediction, risk band, and threshold.

## Streamlit Dashboard

Pages include:

1. Executive Overview
2. Customer Risk
3. Batch Prediction
4. SHAP Explainability
5. Model Monitoring

Continuous automated production drift detection is not yet implemented; the monitoring page documents current coverage and limitations.

## Project Structure

~~~text
enterprise-ml-platform/
|-- api/
|-- dashboard/
|   |-- app.py
|   `-- pages/
|-- models/
|   |-- champion_model.joblib
|   |-- feature_names.json
|   |-- model_metadata.json
|   `-- shap/
|-- spark_jobs/
|-- src/
|-- training/
|-- tests/
|-- requirements.txt
|-- requirements-spark.txt
|-- requirements-dev.txt
`-- pytest.ini
~~~

## Testing

The application test suite contains 16 tests covering prediction logic and API behavior.

~~~text
16 passed
1 warning
~~~

Run:

~~~bash
python -m pytest -q tests/
~~~

The warning is a Starlette/AnyIO deprecation warning and is not a test failure. Spark/HDFS helper scripts are separate because they require a running HDFS environment.

## Local Setup

~~~bash
git clone https://github.com/Harshithpatali/enterprise-ml-platform.git
cd enterprise-ml-platform
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

For development:

~~~bash
pip install -r requirements-dev.txt
~~~

For Spark/Hadoop:

~~~bash
pip install -r requirements-spark.txt
~~~

Run the API:

~~~bash
uvicorn api.main:app --reload
~~~

Run Streamlit in another terminal:

~~~bash
streamlit run dashboard/app.py
~~~

## Deployment

The intended deployment architecture is FastAPI on Render and Streamlit on Streamlit Community Cloud.

~~~text
GitHub
  |
  +--> Render: FastAPI API
  |             ^
  |             |
  +--> Streamlit Cloud: Dashboard
~~~

For deployment, the dashboard should read the API URL from an environment variable rather than hard-coding localhost.

~~~python
import os
API_URL = os.getenv("API_URL", "http://localhost:8000")
~~~

## Production-Oriented Practices

- Temporal feature construction
- Explicit prediction horizon
- Validation-based model selection
- Locked test evaluation
- Persisted preprocessing/model pipeline
- Feature schema validation
- Batch inference
- API health endpoint
- SHAP explainability
- Automated unit/API tests
- Separate runtime and Spark dependencies
- Git-based versioning
- Documented modeling limitations

## Roadmap

- MLflow experiment tracking and model registry
- Automated data/model drift monitoring
- Scheduled retraining
- Data quality contracts
- Airflow orchestration
- PostgreSQL feature serving
- Authentication and authorization
- API rate limiting
- CI/CD deployment gates
- Model calibration monitoring
- Active-customer cohort modeling

## Technology Stack

Data Engineering: Python, Hadoop HDFS, PySpark, Parquet

Machine Learning: scikit-learn, XGBoost, SHAP, Pandas, NumPy

Serving: FastAPI, Uvicorn, Pydantic

Analytics/UI: Streamlit, Plotly

Engineering: Pytest, Git, GitHub

## Author

**Harshith Devaraja**

Machine Learning / Data Science portfolio project focused on production-oriented ML systems, analytics, experimentation, and applied machine learning.

## Disclaimer

This is a portfolio/engineering project based on the public Olist Brazilian E-Commerce dataset. Model metrics are dataset-specific and should not be interpreted as production performance without validation on representative business data.