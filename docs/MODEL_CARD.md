# Model Card — Enterprise Churn Intelligence Platform

## Intended use

Demonstrate an end-to-end analytics and machine-learning workflow for predicting whether a customer will have no completed purchase during the next 90 days, using an Olist-derived customer snapshot. The project is a portfolio demonstration and research artifact, not a validated production CRM policy.

## Data and target

- Source: Brazilian Olist e-commerce data.
- Observation window: 180 days before the snapshot.
- Target horizon: 90 days after the snapshot.
- Target: no completed purchase during the forward 90-day window.
- Features are computed at or before the snapshot; the target uses future outcomes.

## Methods and architecture

HDFS ingestion, PySpark Bronze/Silver/Gold processing, temporal feature construction, model comparison across Logistic Regression, Random Forest and XGBoost, SHAP explanations, FastAPI serving and Streamlit monitoring interface.

## Reported metrics and caveats

The README reports a locked-test PR-AUC of 0.991381 and ROC-AUC of 0.879110 for XGBoost, with 100% churn recall. However, the evaluation population has approximately 96.1% churn prevalence. The README's modeling audit also reports that an always-churn rule already obtains approximately 0.963 precision and 0.981 F1. The incremental value is therefore much more limited than the headline scores suggest; the model's reported retained-customer identification is more informative.

Do not present the headline PR-AUC or F1 without the prevalence, trivial baseline and minority-class results.

## Known limitations

- The high churn prevalence reflects the selected snapshot cohort and limits generalization to active-customer populations.
- The model's retained-customer recall is not equivalent to incremental retention or revenue.
- Thresholds are tied to the documented validation procedure.
- SHAP describes model behavior, not causal drivers.
- A prospective CRM experiment is needed to establish business lift.

## Business interpretation

The platform demonstrates a governed data path from raw orders to a temporal customer dataset and a served prediction. It can help analysts inspect churn-risk ranking and model explanations, but campaign decisions require a better-defined active-customer cohort, stronger baseline comparisons and prospective measurement.

## Reproducibility

```bash
pip install -r requirements-dev.txt
pytest -q tests/
uvicorn api.main:app --reload
streamlit run dashboard/app.py
```

Follow the README for the Hadoop/Spark data preparation sequence and local environment prerequisites.