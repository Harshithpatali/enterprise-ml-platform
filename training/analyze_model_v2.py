import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession


# ============================================================
# CONFIGURATION
# ============================================================

ML_PATH = "hdfs:///data/gold/olist/ml_dataset_v2"

MODEL_PATH = "models/champion_model.joblib"
METADATA_PATH = "models/model_metadata.json"

OUTPUT_DIR = Path("models/shap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 5000
RANDOM_STATE = 42


# ============================================================
# LOAD METADATA
# ============================================================

print("=" * 60)
print("LOADING MODEL METADATA")
print("=" * 60)

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

FEATURES = metadata["feature_names"]
TARGET = metadata["target"]

print(f"Champion model : {metadata['champion_model']}")
print(f"Target         : {TARGET}")
print(f"Threshold      : {metadata['threshold']}")
print(f"Features       : {len(FEATURES)}")


# ============================================================
# LEAKAGE CHECK
# ============================================================

LEAKAGE_COLUMNS = {
    "churn_90d",
    "future_orders_90d",
    "customer_unique_id",
    "observation_date",
    "dataset_split",
}

bad_features = set(FEATURES).intersection(LEAKAGE_COLUMNS)

if bad_features:
    raise ValueError(
        f"LEAKAGE / IDENTIFIER COLUMNS FOUND IN FEATURES: {bad_features}"
    )

print("\nLeakage check: PASSED")


# ============================================================
# LOAD PIPELINE
# ============================================================

print("\n" + "=" * 60)
print("LOADING CHAMPION PIPELINE")
print("=" * 60)

pipeline = joblib.load(MODEL_PATH)

print(f"Pipeline type: {type(pipeline).__name__}")
print(f"Pipeline steps: {list(pipeline.named_steps.keys())}")


# ============================================================
# EXTRACT XGBOOST MODEL
# ============================================================

if "model" not in pipeline.named_steps:
    raise ValueError(
        "Expected a 'model' step in the saved pipeline."
    )

xgb_model = pipeline.named_steps["model"]

print(f"Underlying model: {type(xgb_model).__name__}")


# ============================================================
# EXTRACT PREPROCESSOR
# ============================================================

if "preprocessor" not in pipeline.named_steps:
    raise ValueError(
        "Expected a 'preprocessor' step in the saved pipeline."
    )

preprocessor = pipeline.named_steps["preprocessor"]


# ============================================================
# START SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("OlistChurnSHAP")
    .master("local[*]")
    .config("spark.sql.ansi.enabled", "true")
    .getOrCreate()
)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\n" + "=" * 60)
print("LOADING TEST DATA")
print("=" * 60)

test_path = f"{ML_PATH}/dataset_split=test"

test_df = spark.read.parquet(test_path)

test_count = test_df.count()

print(f"Test rows: {test_count}")

missing_features = [
    feature
    for feature in FEATURES
    if feature not in test_df.columns
]

if missing_features:
    raise ValueError(
        f"Missing features from test dataset: {missing_features}"
    )


# ============================================================
# CONVERT TO PANDAS
# ============================================================

pdf = (
    test_df
    .select(FEATURES + [TARGET])
    .toPandas()
)

print(f"Pandas shape: {pdf.shape}")


# ============================================================
# PREPARE FEATURES
# ============================================================

X = pdf[FEATURES].copy()
y = pdf[TARGET].copy()

for col in FEATURES:
    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

print(
    f"Missing feature values before pipeline: "
    f"{int(X.isna().sum().sum())}"
)


# ============================================================
# SAMPLE TEST DATA
# ============================================================

sample_size = min(
    SAMPLE_SIZE,
    len(X)
)

X_sample = X.sample(
    n=sample_size,
    random_state=RANDOM_STATE
)

y_sample = y.loc[X_sample.index]

print(f"SHAP sample size: {len(X_sample)}")


# ============================================================
# APPLY EXACT TRAINING PREPROCESSOR
# ============================================================

print("\n" + "=" * 60)
print("APPLYING TRAINING PREPROCESSOR")
print("=" * 60)

X_transformed = preprocessor.transform(
    X_sample
)

X_transformed = np.asarray(
    X_transformed
)

print(
    f"Transformed shape: "
    f"{X_transformed.shape}"
)

if X_transformed.shape[1] != len(FEATURES):
    raise ValueError(
        "Number of transformed features does not match "
        "the original feature list."
    )


# ============================================================
# SHAP EXPLAINER
# ============================================================

print("\n" + "=" * 60)
print("CALCULATING SHAP VALUES")
print("=" * 60)

explainer = shap.TreeExplainer(
    xgb_model
)

shap_values = explainer.shap_values(
    X_transformed
)


# ============================================================
# HANDLE SHAP OUTPUT
# ============================================================

if isinstance(shap_values, list):
    shap_values_array = shap_values[-1]
else:
    shap_values_array = shap_values

shap_values_array = np.asarray(
    shap_values_array
)

print(
    f"SHAP shape: "
    f"{shap_values_array.shape}"
)


# ============================================================
# GLOBAL SHAP IMPORTANCE
# ============================================================

mean_abs_shap = np.abs(
    shap_values_array
).mean(axis=0)

importance_df = pd.DataFrame(
    {
        "feature": FEATURES,
        "mean_abs_shap": mean_abs_shap,
    }
).sort_values(
    "mean_abs_shap",
    ascending=False
)

importance_df["importance_pct"] = (
    importance_df["mean_abs_shap"]
    / importance_df["mean_abs_shap"].sum()
    * 100
)

importance_path = (
    OUTPUT_DIR /
    "shap_feature_importance.csv"
)

importance_df.to_csv(
    importance_path,
    index=False
)


# ============================================================
# PRINT TOP FEATURES
# ============================================================

print("\n" + "=" * 60)
print("TOP SHAP FEATURES")
print("=" * 60)

print(
    importance_df
    .head(20)
    .to_string(index=False)
)


# ============================================================
# SHAP SUMMARY PLOT
# ============================================================

plt.figure()

shap.summary_plot(
    shap_values_array,
    X_transformed,
    feature_names=FEATURES,
    show=False
)

plt.tight_layout()

summary_path = (
    OUTPUT_DIR /
    "shap_summary.png"
)

plt.savefig(
    summary_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SHAP BAR PLOT
# ============================================================

plt.figure()

shap.summary_plot(
    shap_values_array,
    X_transformed,
    feature_names=FEATURES,
    plot_type="bar",
    show=False
)

plt.tight_layout()

bar_path = (
    OUTPUT_DIR /
    "shap_bar.png"
)

plt.savefig(
    bar_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE RAW SHAP VALUES
# ============================================================

shap_df = pd.DataFrame(
    shap_values_array,
    columns=FEATURES
)

shap_values_path = (
    OUTPUT_DIR /
    "shap_values_sample.csv"
)

shap_df.to_csv(
    shap_values_path,
    index=False
)


# ============================================================
# HIGH-RISK CUSTOMER SAMPLE
# ============================================================

probabilities = pipeline.predict_proba(
    X_sample
)[:, 1]

customer_examples = X_sample.copy()

customer_examples["actual_churn"] = (
    y_sample.values
)

customer_examples["predicted_probability"] = (
    probabilities
)

customer_examples["predicted_churn"] = (
    probabilities >= metadata["threshold"]
).astype(int)

customer_examples = (
    customer_examples
    .sort_values(
        "predicted_probability",
        ascending=False
    )
)

examples_path = (
    OUTPUT_DIR /
    "high_risk_customers_sample.csv"
)

customer_examples.head(100).to_csv(
    examples_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("SHAP ANALYSIS COMPLETE")
print("=" * 60)

print(f"Feature importance : {importance_path}")
print(f"Summary plot       : {summary_path}")
print(f"Bar plot           : {bar_path}")
print(f"SHAP values        : {shap_values_path}")
print(f"High-risk sample   : {examples_path}")

print("\nTop 10 features:")

for _, row in importance_df.head(10).iterrows():

    print(
        f"{row['feature']:<35}"
        f"{row['mean_abs_shap']:.6f}"
        f" ({row['importance_pct']:.2f}%)"
    )

spark.stop()