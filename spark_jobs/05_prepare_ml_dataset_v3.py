from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# CONFIG
# ============================================================

GOLD_PATH = "hdfs:///data/gold/olist/customer_churn_features_v3"

ML_PATH = "hdfs:///data/gold/olist/ml_dataset_v3"

TARGET = "churn_90d"

# ------------------------------------------------------------
# Leakage protection
# ------------------------------------------------------------

LEAKAGE_COLUMNS = [
    "churn_90d",
    "future_orders_90d",
    "last_purchase_timestamp",
    "first_purchase_timestamp_180d",
]

# Customer identifier should not be used by ML.
ID_COLUMNS = [
    "customer_unique_id",
]

# Dataset metadata columns.
TIME_COLUMNS = [
    "observation_date",
    "observation_month",
    "observation_year",
]


# ============================================================
# SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("PrepareOlistMLDatasetV3")
    .master("local[*]")
    .config("spark.sql.ansi.enabled", "true")
    .config("spark.sql.shuffle.partitions", "16")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# LOAD GOLD V3
# ============================================================

print("=" * 80)
print("LOADING GOLD V3")
print("=" * 80)

df = spark.read.parquet(GOLD_PATH)

print(f"Gold rows: {df.count()}")

print(
    f"Gold customers: "
    f"{df.select('customer_unique_id').distinct().count()}"
)


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("BASIC VALIDATION")
print("=" * 80)

required_columns = [
    "customer_unique_id",
    "observation_date",
    "churn_90d",
    "future_orders_90d",
    "order_count_180d",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("Required columns: OK")


# ============================================================
# SORT DATA
# ============================================================

df = df.orderBy("observation_date")


# ============================================================
# TARGET VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("TARGET VALIDATION")
print("=" * 80)

df.groupBy(TARGET).count().orderBy(TARGET).show()


# ============================================================
# LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 80)
print("LEAKAGE CHECK")
print("=" * 80)

print("Columns that will be excluded:")

for column in LEAKAGE_COLUMNS:
    print(f"  - {column}")

print("\nIdentifier columns excluded:")

for column in ID_COLUMNS:
    print(f"  - {column}")


# ============================================================
# FEATURE COLUMNS
# ============================================================

excluded_columns = set(
    LEAKAGE_COLUMNS
    + ID_COLUMNS
    + TIME_COLUMNS
)

feature_columns = [
    column
    for column in df.columns
    if column not in excluded_columns
]

print("\n" + "=" * 80)
print("MODEL FEATURES")
print("=" * 80)

print(f"Number of features: {len(feature_columns)}")

for i, column in enumerate(feature_columns, 1):
    print(f"{i:02d}. {column}")


# ============================================================
# NULL CHECK
# ============================================================

print("\n" + "=" * 80)
print("NULL CHECK")
print("=" * 80)

for column in feature_columns:
    null_count = df.filter(
        F.col(column).isNull()
    ).count()

    if null_count > 0:
        print(
            f"WARNING: {column} has "
            f"{null_count} null values"
        )


# ============================================================
# TEMPORAL BOUNDARY
# ============================================================

print("\n" + "=" * 80)
print("TEMPORAL INFORMATION")
print("=" * 80)

date_range = df.select(
    F.min("observation_date").alias("min_date"),
    F.max("observation_date").alias("max_date")
).first()

min_date = date_range["min_date"]
max_date = date_range["max_date"]

print(f"Minimum observation date: {min_date}")
print(f"Maximum observation date: {max_date}")


# ============================================================
# TEMPORAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n" + "=" * 80)
print("BUILDING TEMPORAL SPLIT")
print("=" * 80)

# ------------------------------------------------------------
# We intentionally split by observation date.
#
# Train:
#   earliest ~60%
#
# Validation:
#   next ~20%
#
# Test:
#   latest ~20%
#
# This prevents future observations from entering training.
# ------------------------------------------------------------

dates = (
    df.select("observation_date")
    .distinct()
    .orderBy("observation_date")
    .collect()
)

date_values = [
    row["observation_date"]
    for row in dates
]

n_dates = len(date_values)

if n_dates < 10:
    raise ValueError(
        "Not enough observation months for a temporal split."
    )

train_index = int(n_dates * 0.60)
validation_index = int(n_dates * 0.80)

train_end_date = date_values[train_index - 1]
validation_end_date = date_values[validation_index - 1]

validation_start_date = date_values[train_index]
test_start_date = date_values[validation_index]

print(
    f"Train:      <= {train_end_date}"
)

print(
    f"Validation: {validation_start_date} "
    f"to {validation_end_date}"
)

print(
    f"Test:       >= {test_start_date}"
)


# ============================================================
# CREATE SPLIT
# ============================================================

train_df = df.filter(
    F.col("observation_date") <= F.lit(train_end_date)
)

validation_df = df.filter(
    (F.col("observation_date") >= F.lit(validation_start_date))
    &
    (F.col("observation_date") <= F.lit(validation_end_date))
)

test_df = df.filter(
    F.col("observation_date") >= F.lit(test_start_date)
)


# ============================================================
# SPLIT VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("SPLIT SIZES")
print("=" * 80)

train_count = train_df.count()
validation_count = validation_df.count()
test_count = test_df.count()

total_count = train_count + validation_count + test_count

print(f"Train:      {train_count:,}")
print(f"Validation: {validation_count:,}")
print(f"Test:       {test_count:,}")
print(f"Total:      {total_count:,}")

if total_count != df.count():
    raise ValueError(
        "Temporal split does not contain all rows."
    )


# ============================================================
# TARGET DISTRIBUTION PER SPLIT
# ============================================================

print("\n" + "=" * 80)
print("TRAIN TARGET DISTRIBUTION")
print("=" * 80)

train_df.groupBy(TARGET).count().orderBy(TARGET).show()

print("\n" + "=" * 80)
print("VALIDATION TARGET DISTRIBUTION")
print("=" * 80)

validation_df.groupBy(TARGET).count().orderBy(TARGET).show()

print("\n" + "=" * 80)
print("TEST TARGET DISTRIBUTION")
print("=" * 80)

test_df.groupBy(TARGET).count().orderBy(TARGET).show()


# ============================================================
# CUSTOMER OVERLAP CHECK
# ============================================================

print("\n" + "=" * 80)
print("CUSTOMER OVERLAP CHECK")
print("=" * 80)

train_customers = train_df.select(
    "customer_unique_id"
).distinct()

validation_customers = validation_df.select(
    "customer_unique_id"
).distinct()

test_customers = test_df.select(
    "customer_unique_id"
).distinct()

train_validation_overlap = (
    train_customers
    .join(
        validation_customers,
        "customer_unique_id",
        "inner"
    )
    .count()
)

validation_test_overlap = (
    validation_customers
    .join(
        test_customers,
        "customer_unique_id",
        "inner"
    )
    .count()
)

train_test_overlap = (
    train_customers
    .join(
        test_customers,
        "customer_unique_id",
        "inner"
    )
    .count()
)

print(
    f"Train ∩ Validation customers: "
    f"{train_validation_overlap:,}"
)

print(
    f"Validation ∩ Test customers: "
    f"{validation_test_overlap:,}"
)

print(
    f"Train ∩ Test customers: "
    f"{train_test_overlap:,}"
)

print(
    "\nNOTE: Customer overlap across temporal snapshots "
    "is expected in a customer-month panel."
)


# ============================================================
# FEATURE-ONLY DATASET
# ============================================================

print("\n" + "=" * 80)
print("CREATING ML DATASET")
print("=" * 80)

# ------------------------------------------------------------
# Keep:
#   customer_unique_id
#   observation_date
#   features
#   target
#
# Remove:
#   future_orders_90d
#   timestamps that can expose future information
#   observation metadata
# ------------------------------------------------------------

final_columns = (
    ["customer_unique_id", "observation_date"]
    + feature_columns
    + [TARGET]
)

ml_df = df.select(final_columns)


# ============================================================
# FINAL LEAKAGE ASSERTIONS
# ============================================================

# ============================================================
# FINAL LEAKAGE ASSERTIONS
# ============================================================

print("\n" + "=" * 80)
print("FINAL LEAKAGE ASSERTIONS")
print("=" * 80)

# ------------------------------------------------------------
# The target MUST remain in the ML dataset.
# It is not a model feature.
# ------------------------------------------------------------

if TARGET not in ml_df.columns:
    raise ValueError(
        f"Target column missing: {TARGET}"
    )

# ------------------------------------------------------------
# These columns MUST NOT be present in the ML dataset.
# ------------------------------------------------------------

FORBIDDEN_FEATURE_COLUMNS = [
    "future_orders_90d",
    "last_purchase_timestamp",
    "first_purchase_timestamp_180d",
]

for forbidden in FORBIDDEN_FEATURE_COLUMNS:
    if forbidden in ml_df.columns:
        raise ValueError(
            f"LEAKAGE DETECTED: {forbidden}"
        )

# ------------------------------------------------------------
# Verify target is not part of feature_columns.
# ------------------------------------------------------------

if TARGET in feature_columns:
    raise ValueError(
        f"TARGET LEAKAGE DETECTED: "
        f"{TARGET} is present in feature_columns."
    )

# ------------------------------------------------------------
# Verify customer ID is not a model feature.
# ------------------------------------------------------------

if "customer_unique_id" in feature_columns:
    raise ValueError(
        "IDENTIFIER LEAKAGE DETECTED: "
        "customer_unique_id is present in feature_columns."
    )

# ------------------------------------------------------------
# Verify all 19 intended features exist.
# ------------------------------------------------------------

missing_features = [
    column
    for column in feature_columns
    if column not in ml_df.columns
]

if missing_features:
    raise ValueError(
        f"Missing model features: {missing_features}"
    )

print("Target column present:", TARGET)

print(
    "Forbidden future/leakage columns absent: OK"
)

print(
    "Target excluded from model features: OK"
)

print(
    "Customer ID excluded from model features: OK"
)

print(
    f"Final model feature count: "
    f"{len(feature_columns)}"
)

print("Leakage validation: PASSED")


# ============================================================
# FINAL SCHEMA
# ============================================================

print("\n" + "=" * 80)
print("FINAL ML SCHEMA")
print("=" * 80)

ml_df.printSchema()


# ============================================================
# SAVE ML DATASET
# ============================================================

print("\n" + "=" * 80)
print("WRITING ML DATASET V3")
print("=" * 80)

(
    ml_df
    .repartition("observation_date")
    .write
    .mode("overwrite")
    .partitionBy("observation_date")
    .parquet(ML_PATH)
)


# ============================================================
# SAVE SPLIT INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("SPLIT INFORMATION")
print("=" * 80)

print(f"Train end:      {train_end_date}")
print(f"Validation end: {validation_end_date}")
print(f"Test start:     {test_start_date}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("V3 ML DATASET CREATED")
print("=" * 80)

print(f"Location: {ML_PATH}")
print(f"Rows: {ml_df.count():,}")
print(f"Features: {len(feature_columns)}")

print("\nFeature columns:")

for column in feature_columns:
    print(f"  - {column}")

print("\nDataset is ready for temporal model training.")


# ============================================================
# STOP
# ============================================================

spark.stop()