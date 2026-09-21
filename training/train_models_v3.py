import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

ML_PATH = "hdfs:///data/gold/olist/ml_dataset_v3"

MODEL_DIR = "models/v3"

TARGET = "churn_90d"

FEATURE_COLUMNS = [
    "order_count_180d",
    "revenue_180d",
    "payment_value_180d",
    "freight_value_180d",
    "item_count_180d",
    "avg_payment_type_count_180d",
    "avg_installments_180d",
    "avg_review_score_180d",
    "review_count_180d",
    "avg_delivery_delay_180d",
    "late_delivery_rate_180d",
    "unique_products_180d",
    "unique_categories_180d",
    "unique_sellers_180d",
    "recency_days",
    "customer_lifetime_days",
    "avg_order_value_180d",
    "orders_per_month_180d",
    "revenue_per_month_180d",
]

TRAIN_END = "2017-08-31"
VALIDATION_START = "2017-09-30"
VALIDATION_END = "2017-12-31"
TEST_START = "2018-01-31"


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("LOADING V3 ML DATASET")
print("=" * 80)

# PyArrow can read HDFS through Spark, so use PySpark here.
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("OlistChurnModelTrainingV3")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "16")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

df_spark = spark.read.parquet(ML_PATH)

df = df_spark.select(
    ["customer_unique_id", "observation_date"]
    + FEATURE_COLUMNS
    + [TARGET]
).toPandas()

spark.stop()

print(f"Rows: {len(df):,}")
print(f"Features: {len(FEATURE_COLUMNS)}")


# ============================================================
# DATE CONVERSION
# ============================================================

df["observation_date"] = pd.to_datetime(
    df["observation_date"]
)


# ============================================================
# TEMPORAL SPLIT
# ============================================================

print("\n" + "=" * 80)
print("TEMPORAL SPLIT")
print("=" * 80)

train = df[
    df["observation_date"]
    <= pd.Timestamp(TRAIN_END)
].copy()

validation = df[
    (df["observation_date"] >= pd.Timestamp(VALIDATION_START))
    &
    (df["observation_date"] <= pd.Timestamp(VALIDATION_END))
].copy()

test = df[
    df["observation_date"]
    >= pd.Timestamp(TEST_START)
].copy()

print(f"Train:      {len(train):,}")
print(f"Validation: {len(validation):,}")
print(f"Test:       {len(test):,}")


# ============================================================
# X / Y
# ============================================================

X_train = train[FEATURE_COLUMNS]
y_train = train[TARGET].astype(int)

X_validation = validation[FEATURE_COLUMNS]
y_validation = validation[TARGET].astype(int)

X_test = test[FEATURE_COLUMNS]
y_test = test[TARGET].astype(int)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("CLASS DISTRIBUTION")
print("=" * 80)

print("Train:")
print(y_train.value_counts())

print("\nValidation:")
print(y_validation.value_counts())

print("\nTest:")
print(y_test.value_counts())


# ============================================================
# CLASS WEIGHT
# ============================================================

negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

scale_pos_weight = negative_count / positive_count

# Because churn=1 is the majority class, the conventional
# XGBoost scale_pos_weight is not appropriate here.
#
# Instead, we weight the minority retained class (0).

minority_weight = positive_count / negative_count

sample_weights = np.where(
    y_train == 0,
    minority_weight,
    1.0,
)

print("\n" + "=" * 80)
print("CLASS WEIGHTING")
print("=" * 80)

print(
    f"Retained class weight: "
    f"{minority_weight:.4f}"
)

print(
    f"Churn class weight: "
    f"1.0000"
)


# ============================================================
# PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median"),
                    ),
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                ]
            ),
            FEATURE_COLUMNS,
        )
    ]
)


# ============================================================
# MODELS
# ============================================================

models = {}

# ------------------------------------------------------------
# Logistic Regression
# ------------------------------------------------------------

models["logistic_regression"] = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                C=1.0,
                solver="lbfgs",
            ),
        ),
    ]
)


# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------

models["random_forest"] = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=20,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


# ------------------------------------------------------------
# XGBoost
# ------------------------------------------------------------

models["xgboost"] = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            XGBClassifier(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                objective="binary:logistic",
                eval_metric="aucpr",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


# ------------------------------------------------------------
# LightGBM
# ------------------------------------------------------------

if LIGHTGBM_AVAILABLE:

    models["lightgbm"] = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.03,
                    max_depth=5,
                    num_leaves=31,
                    min_child_samples=30,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary",
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    )

else:
    print(
        "\nLightGBM is not installed. "
        "Skipping LightGBM."
    )


# ------------------------------------------------------------
# Neural Network
# ------------------------------------------------------------

models["neural_network"] = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=0.0005,
                batch_size=512,
                learning_rate_init=0.001,
                max_iter=150,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=10,
                random_state=42,
            ),
        ),
    ]
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def evaluate_model(
    name,
    model,
    X,
    y,
    split_name,
    sample_weight=None,
):
    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {
        "model": name,
        "split": split_name,
        "pr_auc": average_precision_score(
            y,
            probabilities
        ),
        "roc_auc": roc_auc_score(
            y,
            probabilities
        ),
        "precision": precision_score(
            y,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y,
            predictions,
            zero_division=0,
        ),
    }

    return metrics, probabilities


# ============================================================
# TRAIN
# ============================================================

results = []

validation_probabilities = {}
test_probabilities = {}

print("\n" + "=" * 80)
print("MODEL TRAINING")
print("=" * 80)


for name, model in models.items():

    print("\n" + "-" * 80)
    print(f"TRAINING: {name}")
    print("-" * 80)

    # --------------------------------------------------------
    # Fit
    # --------------------------------------------------------

    if name in [
        "logistic_regression",
        "random_forest",
        "neural_network",
    ]:

        model.fit(
            X_train,
            y_train,
        )

    else:

        # Tree boosting models receive the explicit
        # minority-class weighting.
        model.fit(
            X_train,
            y_train,
            model__sample_weight=sample_weights,
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_metrics, val_prob = evaluate_model(
        name,
        model,
        X_validation,
        y_validation,
        "validation",
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_metrics, test_prob = evaluate_model(
        name,
        model,
        X_test,
        y_test,
        "test",
    )

    validation_probabilities[name] = val_prob
    test_probabilities[name] = test_prob

    results.append(val_metrics)
    results.append(test_metrics)

    print(
        f"Validation PR-AUC: "
        f"{val_metrics['pr_auc']:.6f}"
    )

    print(
        f"Validation ROC-AUC: "
        f"{val_metrics['roc_auc']:.6f}"
    )

    print(
        f"Test PR-AUC: "
        f"{test_metrics['pr_auc']:.6f}"
    )

    print(
        f"Test ROC-AUC: "
        f"{test_metrics['roc_auc']:.6f}"
    )


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    f"{MODEL_DIR}/model_results.csv",
    index=False,
)

print("\n" + "=" * 80)
print("MODEL RESULTS")
print("=" * 80)

print(
    results_df[
        [
            "model",
            "split",
            "pr_auc",
            "roc_auc",
            "precision",
            "recall",
            "f1",
        ]
    ].to_string(index=False)
)


# ============================================================
# SELECT CHAMPION
# ============================================================

validation_results = results_df[
    results_df["split"] == "validation"
].copy()

validation_results = validation_results.sort_values(
    "pr_auc",
    ascending=False,
)

champion_name = validation_results.iloc[0]["model"]

print("\n" + "=" * 80)
print("CHAMPION MODEL")
print("=" * 80)

print(
    f"Selected by validation PR-AUC: "
    f"{champion_name}"
)


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

print("\n" + "=" * 80)
print("THRESHOLD OPTIMIZATION")
print("=" * 80)

champion_val_prob = validation_probabilities[
    champion_name
]

threshold_results = []

for threshold in np.arange(
    0.01,
    1.00,
    0.01,
):

    predictions = (
        champion_val_prob >= threshold
    ).astype(int)

    threshold_results.append(
        {
            "threshold": threshold,
            "precision": precision_score(
                y_validation,
                predictions,
                zero_division=0,
            ),
            "recall": recall_score(
                y_validation,
                predictions,
                zero_division=0,
            ),
            "f1": f1_score(
                y_validation,
                predictions,
                zero_division=0,
            ),
        }
    )


threshold_df = pd.DataFrame(
    threshold_results
)

best_threshold_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]

best_threshold = float(
    best_threshold_row["threshold"]
)

threshold_df.to_csv(
    f"{MODEL_DIR}/threshold_results.csv",
    index=False,
)

print(
    f"Best validation F1 threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Validation precision: "
    f"{best_threshold_row['precision']:.6f}"
)

print(
    f"Validation recall: "
    f"{best_threshold_row['recall']:.6f}"
)

print(
    f"Validation F1: "
    f"{best_threshold_row['f1']:.6f}"
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

champion_test_prob = test_probabilities[
    champion_name
]

test_predictions = (
    champion_test_prob >= best_threshold
).astype(int)

final_test_pr_auc = average_precision_score(
    y_test,
    champion_test_prob,
)

final_test_roc_auc = roc_auc_score(
    y_test,
    champion_test_prob,
)

final_test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0,
)

final_test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0,
)

final_test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0,
)

cm = confusion_matrix(
    y_test,
    test_predictions,
)


# ============================================================
# FINAL METRICS
# ============================================================

print("\n" + "=" * 80)
print("FINAL TEST RESULTS")
print("=" * 80)

print(
    f"Champion:       {champion_name}"
)

print(
    f"Threshold:      {best_threshold:.2f}"
)

print(
    f"PR-AUC:         {final_test_pr_auc:.6f}"
)

print(
    f"ROC-AUC:        {final_test_roc_auc:.6f}"
)

print(
    f"Precision:      {final_test_precision:.6f}"
)

print(
    f"Recall:         {final_test_recall:.6f}"
)

print(
    f"F1:             {final_test_f1:.6f}"
)

print("\nConfusion Matrix:")

print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        digits=6,
        zero_division=0,
    )
)


# ============================================================
# SAVE CHAMPION
# ============================================================

champion_model = models[
    champion_name
]

champion_path = (
    f"{MODEL_DIR}/champion_model.joblib"
)

joblib.dump(
    champion_model,
    champion_path,
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {
    "version": "v3",
    "dataset": ML_PATH,
    "champion_model": champion_name,
    "target": TARGET,
    "selection_metric": "validation_pr_auc",
    "threshold_selection_metric": "validation_f1",
    "threshold": best_threshold,
    "train_end": TRAIN_END,
    "validation_start": VALIDATION_START,
    "validation_end": VALIDATION_END,
    "test_start": TEST_START,
    "feature_count": len(FEATURE_COLUMNS),
    "features": FEATURE_COLUMNS,
    "train_rows": len(train),
    "validation_rows": len(validation),
    "test_rows": len(test),
    "test_metrics": {
        "pr_auc": final_test_pr_auc,
        "roc_auc": final_test_roc_auc,
        "precision": final_test_precision,
        "recall": final_test_recall,
        "f1": final_test_f1,
    },
}


with open(
    f"{MODEL_DIR}/model_metadata.json",
    "w",
) as f:
    json.dump(
        metadata,
        f,
        indent=4,
    )


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

test_predictions_df = test[
    [
        "customer_unique_id",
        "observation_date",
        TARGET,
    ]
].copy()

test_predictions_df[
    "churn_probability"
] = champion_test_prob

test_predictions_df[
    "churn_prediction"
] = test_predictions

test_predictions_df.to_parquet(
    f"{MODEL_DIR}/test_predictions.parquet",
    index=False,
)


# ============================================================
# SAVE FEATURE LIST
# ============================================================

with open(
    f"{MODEL_DIR}/feature_names.json",
    "w",
) as f:
    json.dump(
        FEATURE_COLUMNS,
        f,
        indent=4,
    )


# ============================================================
# SAVE FINAL METRICS
# ============================================================

final_metrics = {
    "champion_model": champion_name,
    "threshold": best_threshold,
    "test_pr_auc": final_test_pr_auc,
    "test_roc_auc": final_test_roc_auc,
    "test_precision": final_test_precision,
    "test_recall": final_test_recall,
    "test_f1": final_test_f1,
    "confusion_matrix": cm.tolist(),
}


with open(
    f"{MODEL_DIR}/final_metrics.json",
    "w",
) as f:
    json.dump(
        final_metrics,
        f,
        indent=4,
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("V3 MODEL TRAINING COMPLETE")
print("=" * 80)

print(
    f"Champion model saved to: "
    f"{champion_path}"
)

print(
    f"Artifacts directory: "
    f"{MODEL_DIR}"
)

print("=" * 80)