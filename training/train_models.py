from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid

from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ML_PATH = "hdfs:///data/gold/olist/ml_dataset_v2"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "churn_90d"

ID_COLUMNS = [
    "customer_unique_id",
    "observation_date",
    "dataset_split",
]


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset():

    print("\nLoading ML dataset from HDFS...")

    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .appName("OlistMLTrainingLoader")
        .config("spark.sql.ansi.enabled", "true")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    df = (
        spark.read
        .parquet(ML_PATH)
    )

    print(
        f"Total Spark rows: {df.count():,}"
    )

    print("\nSchema:")
    df.printSchema()

    pandas_df = df.toPandas()

    spark.stop()

    return pandas_df


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data(df):

    print("\nPreparing modeling data...")

    df = df.copy()

    df = df.sort_values(
        "observation_date"
    ).reset_index(drop=True)

    train_df = df[
        df["dataset_split"] == "train"
    ].copy()

    validation_df = df[
        df["dataset_split"] == "validation"
    ].copy()

    test_df = df[
        df["dataset_split"] == "test"
    ].copy()

    print(
        f"Train rows:       {len(train_df):,}"
    )

    print(
        f"Validation rows:  {len(validation_df):,}"
    )

    print(
        f"Test rows:        {len(test_df):,}"
    )

    print("\nTarget distribution:")

    for name, dataset in [
        ("TRAIN", train_df),
        ("VALIDATION", validation_df),
        ("TEST", test_df),
    ]:

        print(
            f"\n{name}"
        )

        print(
            dataset[TARGET]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        print(
            "Churn rate:",
            round(
                dataset[TARGET].mean(),
                4,
            ),
        )

    return (
        train_df,
        validation_df,
        test_df,
    )


# ============================================================
# FEATURE SELECTION
# ============================================================

def build_features(
    train_df,
    validation_df,
    test_df,
):
    print("\nBuilding model features...")

    # ========================================================
    # TARGET + LEAKAGE PROTECTION
    # ========================================================
    TARGET = "churn_90d"

    LEAKAGE_COLUMNS = [
        "churn_90d",
        "future_orders_90d",
    ]

    # Build the feature list from the complete dataframe schema.
    # Target and future-looking variables are explicitly excluded.
    df = pd.concat(
        [train_df, validation_df, test_df],
        axis=0,
        ignore_index=True,
    )

    feature_cols = [
        c for c in df.columns
        if c not in LEAKAGE_COLUMNS
        and c not in [
            "customer_unique_id",
            "observation_date",
            "dataset_split",
        ]
    ]

    # --------------------------------------------------------
    # HARD TARGET-LEAKAGE SAFETY CHECK
    # --------------------------------------------------------
    for forbidden in [
        "churn_90d",
        "future_orders_90d",
    ]:
        if forbidden in feature_cols:
            raise ValueError(
                f"TARGET LEAKAGE DETECTED: {forbidden} "
                f"is present in feature_cols"
            )

    # Additional defensive check for any unexpected target-like
    # columns that may have been introduced into the dataset.
    forbidden_present = set(LEAKAGE_COLUMNS).intersection(feature_cols)
    if forbidden_present:
        raise ValueError(
            "TARGET LEAKAGE DETECTED: "
            f"{sorted(forbidden_present)} are present in feature_cols"
        )

    print(
        f"\nCandidate features before type filtering: "
        f"{len(feature_cols)}"
    )

    print("\nSelected candidate features:")
    for feature in feature_cols:
        print(f"  - {feature}")

    # --------------------------------------------------------
    # Construct X/y using the explicitly protected feature list.
    # --------------------------------------------------------
    X_train = train_df[feature_cols].copy()
    X_validation = validation_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    y_train = train_df[TARGET].astype(int)
    y_validation = validation_df[TARGET].astype(int)
    y_test = test_df[TARGET].astype(int)

    # --------------------------------------------------------
    # Current modeling pipeline is numeric-only.
    # Keep only numeric columns while preserving the leakage
    # protection above.
    # --------------------------------------------------------
    numeric_features = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # Run the safety check AGAIN after numeric filtering.
    for forbidden in LEAKAGE_COLUMNS:
        if forbidden in numeric_features:
            raise ValueError(
                f"TARGET LEAKAGE DETECTED: {forbidden} "
                f"is present in numeric_features"
            )

    X_train = X_train[numeric_features]
    X_validation = X_validation[numeric_features]
    X_test = X_test[numeric_features]

    print(
        f"\nNumber of model features: "
        f"{len(numeric_features)}"
    )

    print("\nFinal numeric features:")
    for feature in numeric_features:
        print(f"  - {feature}")

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        numeric_features,
    )


# ============================================================
# PREPROCESSOR
# ============================================================

def create_preprocessor():

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )


# ============================================================
# METRICS
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    dataset_name,
    threshold=0.5,
):

    probabilities = model.predict_proba(
        X
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "dataset": dataset_name,
        "threshold": threshold,
        "roc_auc": roc_auc_score(
            y,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y,
            probabilities,
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

    print(
        f"\n===== {dataset_name} ====="
    )

    for key, value in metrics.items():

        if isinstance(value, float):

            print(
                f"{key:15s}: "
                f"{value:.6f}"
            )

        else:

            print(
                f"{key:15s}: "
                f"{value}"
            )

    print("\nConfusion matrix:")

    print(
        confusion_matrix(
            y,
            predictions,
        )
    )

    return metrics


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def threshold_analysis(
    model,
    X,
    y,
):

    probabilities = model.predict_proba(
        X
    )[:, 1]

    thresholds = np.arange(
        0.05,
        0.96,
        0.05,
    )

    rows = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        rows.append(
            {
                "threshold": threshold,
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
        )

    results = pd.DataFrame(rows)

    results = results.sort_values(
        "f1",
        ascending=False,
    )

    print(
        "\nThreshold analysis:"
    )

    print(
        results.to_string(
            index=False
        )
    )

    return results


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

def train_logistic(
    X_train,
    y_train,
):

    print(
        "\nTraining Logistic Regression..."
    )

    positive = (
        y_train == 1
    ).sum()

    negative = (
        y_train == 0
    ).sum()

    class_weight = {
        0: 1.0,
        1: negative / positive,
    }

    print(
        "Class weights:",
        class_weight,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight=class_weight,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    return pipeline


# ============================================================
# RANDOM FOREST
# ============================================================

def train_random_forest(
    X_train,
    y_train,
):

    print(
        "\nTraining Random Forest..."
    )

    positive = (
        y_train == 1
    ).sum()

    negative = (
        y_train == 0
    ).sum()

    scale = negative / positive

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=20,
                    class_weight={
                        0: 1.0,
                        1: scale,
                    },
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# ============================================================
# XGBOOST GRID SEARCH
# ============================================================

def train_xgboost(
    X_train,
    y_train,
    X_validation,
    y_validation,
):

    print(
        "\nStarting XGBoost hyperparameter search..."
    )

    positive = (
        y_train == 1
    ).sum()

    negative = (
        y_train == 0
    ).sum()

    scale_pos_weight = (
        negative / positive
    )

    print(
        f"scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    parameter_grid = {

        "max_depth": [
            3,
            5,
        ],

        "learning_rate": [
            0.03,
            0.08,
        ],

        "n_estimators": [
            200,
            400,
        ],

        "subsample": [
            0.8,
        ],

        "colsample_bytree": [
            0.8,
        ],
    }

    results = []

    best_model = None
    best_score = -np.inf
    best_params = None

    for params in ParameterGrid(
        parameter_grid
    ):

        print(
            "\nTesting:",
            params,
        )

        model = Pipeline(
            steps=[
                (
                    "preprocessor",
                    create_preprocessor(),
                ),
                (
                    "model",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="aucpr",
                        random_state=42,
                        n_jobs=-1,
                        tree_method="hist",
                        scale_pos_weight=scale_pos_weight,
                        **params,
                    ),
                ),
            ]
        )

        model.fit(
            X_train,
            y_train,
        )

        probabilities = (
            model.predict_proba(
                X_validation
            )[:, 1]
        )

        pr_auc = (
            average_precision_score(
                y_validation,
                probabilities,
            )
        )

        roc_auc = (
            roc_auc_score(
                y_validation,
                probabilities,
            )
        )

        results.append(
            {
                **params,
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
            }
        )

        print(
            f"Validation PR-AUC: "
            f"{pr_auc:.6f}"
        )

        print(
            f"Validation ROC-AUC: "
            f"{roc_auc:.6f}"
        )

        if pr_auc > best_score:

            best_score = pr_auc
            best_model = model
            best_params = params

    results_df = pd.DataFrame(
        results
    ).sort_values(
        "pr_auc",
        ascending=False,
    )

    results_df.to_csv(
        MODEL_DIR
        / "xgboost_grid_results.csv",
        index=False,
    )

    print(
        "\nBest XGBoost parameters:"
    )

    print(
        best_params
    )

    print(
        f"Best validation PR-AUC: "
        f"{best_score:.6f}"
    )

    return (
        best_model,
        best_params,
        results_df,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n============================================"
    )
    print(
        " OLIST CHURN MODEL TRAINING"
    )
    print(
        "============================================"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    (
        train_df,
        validation_df,
        test_df,
    ) = prepare_data(df)

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        feature_names,
    ) = build_features(
        train_df,
        validation_df,
        test_df,
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = {}

    # Logistic Regression
    models[
        "logistic_regression"
    ] = train_logistic(
        X_train,
        y_train,
    )

    # Random Forest
    models[
        "random_forest"
    ] = train_random_forest(
        X_train,
        y_train,
    )

    # XGBoost
    (
        models["xgboost"],
        best_params,
        grid_results,
    ) = train_xgboost(
        X_train,
        y_train,
        X_validation,
        y_validation,
    )

    # --------------------------------------------------------
    # Validation evaluation
    # --------------------------------------------------------

    validation_results = []

    print(
        "\n============================================"
    )
    print(
        " VALIDATION RESULTS"
    )
    print(
        "============================================"
    )

    for name, model in models.items():

        metrics = evaluate_model(
            model,
            X_validation,
            y_validation,
            f"{name}_validation",
        )

        metrics[
            "model"
        ] = name

        validation_results.append(
            metrics
        )

    validation_df_results = pd.DataFrame(
        validation_results
    )

    validation_df_results = (
        validation_df_results
        .sort_values(
            "pr_auc",
            ascending=False,
        )
    )

    validation_df_results.to_csv(
        MODEL_DIR
        / "validation_results.csv",
        index=False,
    )

    print(
        "\nModel comparison:"
    )

    print(
        validation_df_results.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Select champion based on validation PR-AUC
    # --------------------------------------------------------

    champion_name = (
        validation_df_results
        .iloc[0]["model"]
    )

    champion = models[
        champion_name
    ]

    print(
        "\nChampion model:"
    )

    print(
        champion_name
    )

    # --------------------------------------------------------
    # Threshold analysis
    # --------------------------------------------------------

    threshold_results = (
        threshold_analysis(
            champion,
            X_validation,
            y_validation,
        )
    )

    threshold_results.to_csv(
        MODEL_DIR
        / "threshold_analysis.csv",
        index=False,
    )

    # Select threshold with maximum
    # validation F1.
    best_threshold = float(
        threshold_results.iloc[0][
            "threshold"
        ]
    )

    print(
        f"\nSelected validation threshold: "
        f"{best_threshold:.2f}"
    )

    # --------------------------------------------------------
    # Final validation evaluation
    # --------------------------------------------------------

    evaluate_model(
        champion,
        X_validation,
        y_validation,
        "CHAMPION_VALIDATION",
        threshold=best_threshold,
    )

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    print(
        "\n============================================"
    )
    print(
        " FINAL TEST EVALUATION"
    )
    print(
        "============================================"
    )

    test_metrics = evaluate_model(
        champion,
        X_test,
        y_test,
        "CHAMPION_TEST",
        threshold=best_threshold,
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    test_probabilities = (
        champion.predict_proba(
            X_test
        )[:, 1]
    )

    test_predictions = (
        test_probabilities
        >= best_threshold
    ).astype(int)

    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            y_test,
            test_predictions,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Save champion
    # --------------------------------------------------------

    joblib.dump(
        champion,
        MODEL_DIR
        / "champion_model.joblib",
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {

        "champion_model":
            champion_name,

        "target":
            TARGET,

        "selection_metric":
            "validation_pr_auc",

        "threshold":
            best_threshold,

        "feature_names":
            feature_names,

        "xgboost_best_params":
            best_params,

        "validation_results":
            validation_df_results
            .to_dict(
                orient="records"
            ),

        "test_metrics":
            test_metrics,

    }

    with open(
        MODEL_DIR
        / "model_metadata.json",
        "w",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            default=str,
        )

    # --------------------------------------------------------
    # Save feature list
    # --------------------------------------------------------

    with open(
        MODEL_DIR
        / "feature_names.json",
        "w",
    ) as f:

        json.dump(
            feature_names,
            f,
            indent=2,
        )

    print(
        "\n============================================"
    )
    print(
        " TRAINING COMPLETE"
    )
    print(
        "============================================"
    )

    print(
        f"\nChampion: "
        f"{champion_name}"
    )

    print(
        f"Threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        "\nArtifacts:"
    )

    print(
        "models/champion_model.joblib"
    )

    print(
        "models/model_metadata.json"
    )

    print(
        "models/feature_names.json"
    )

    print(
        "models/validation_results.csv"
    )

    print(
        "models/threshold_analysis.csv"
    )

    print(
        "models/xgboost_grid_results.csv"
    )


if __name__ == "__main__":
    main()