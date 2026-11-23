# Enterprise churn platform — experiment and artifact audit

**Purpose:** document what the committed model artifacts support and prevent inflated interpretation of highly imbalanced churn metrics.

## Target and decision

The platform predicts whether a customer has no completed purchase during the next 90 days, using a 180-day observation window and 19 customer-level features. The current model card describes this as a portfolio demonstration, not a validated production CRM policy.

## Why headline metrics need context

The root `models/model_metadata.json` records an XGBoost champion with test PR-AUC 0.991381, ROC-AUC 0.879110, precision 0.988654 and recall 1.0 at threshold 0.20.

The separate `models/v3/model_metadata.json` and `models/v3/final_metrics.json` record a different v3 champion (Logistic Regression) with test PR-AUC 0.993982, ROC-AUC 0.591532, precision 0.992516, recall 0.999961 and F1 0.996225 at threshold 0.01. The v3 confusion matrix is [[6, 1343], [7, 178116]], which implies an exceptionally high positive-class prevalence in that test population.

These artifacts describe different model versions/configurations and must not be combined into one performance claim. Before selecting one as the portfolio's current champion, regenerate a single run manifest tying together the dataset fingerprint, model version, split boundaries, class prevalence, confusion matrix, threshold, metric JSON and serialized model checksum.

## Baseline interpretation

The existing model card and README warn that an always-churn rule already performs strongly because the selected snapshot cohort has very high churn prevalence. Accuracy-like and PR-AUC values must therefore be reported beside:

- positive-class prevalence;
- always-positive and prior-probability baselines;
- confusion matrix and class-specific precision/recall;
- retained-customer identification metrics (the minority business-relevant class);
- fixed-capacity review/retention metrics;
- calibration and threshold-selection policy.

The v3 confusion matrix shows 1,343 negative-class examples among 179,472 test rows. A model that predicts nearly every row as churn may look strong on the majority class while providing limited prioritization value for retained customers.

## Reproducible setup and tests

```bash
python -m venv .venv
# Activate the environment
python -m pip install -r requirements-dev.txt
python -m pytest -q tests/
uvicorn api.main:app --reload
streamlit run dashboard/app.py
```

The HDFS/PySpark data build has separate environment requirements in `requirements-spark.txt`; follow the README for Hadoop setup and the Bronze/Silver/Gold job sequence. The standard API/model test suite does not rebuild the full distributed dataset.

## Recommended next experiment

1. Freeze one model version and dataset snapshot with a single machine-readable run manifest.
2. Recompute an always-churn and prior-probability baseline on exactly the same test rows.
3. Report specificity, negative-class recall, precision/recall at a fixed retention capacity and PR-AUC for the retained class.
4. Verify chronological split and target maturity; retain a later untouched temporal period if available.
5. Compare the current model against the baseline before claiming incremental value.
6. Run a prospective randomized retention campaign before translating risk ranking into incremental retained customers or revenue.

## Business interpretation

This project demonstrates a distributed data-engineering pipeline, a model-serving API and an explainability/monitoring interface. The current evidence does not yet show that the model materially improves retention decisions over a trivial majority-class rule. The business value should be framed as an operational analytics platform with a model requiring further baseline validation, not as proven churn reduction.
