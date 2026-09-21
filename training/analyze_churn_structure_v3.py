from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# CONFIG
# ============================================================

ML_PATH = "hdfs:///data/gold/olist/customer_churn_features_v3"


# ============================================================
# SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("AnalyzeOlistChurnStructureV3")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "16")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# LOAD V3 GOLD DATA
# ============================================================

print("=" * 80)
print("LOADING V3 GOLD DATA")
print("=" * 80)

df = spark.read.parquet(ML_PATH)

print(f"Rows: {df.count()}")
print(
    f"Unique customers: "
    f"{df.select('customer_unique_id').distinct().count()}"
)

print("\nSchema:")
df.printSchema()


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("1. TARGET DISTRIBUTION")
print("=" * 80)

target = (
    df.groupBy("churn_90d")
    .agg(
        F.count("*").alias("rows"),
        F.countDistinct("customer_unique_id").alias("customers")
    )
    .withColumn(
        "percentage",
        F.round(
            F.col("rows") / F.sum("rows").over(
                __import__("pyspark").sql.Window.partitionBy()
            ) * 100,
            4
        )
    )
    .orderBy("churn_90d")
)

target.show(truncate=False)


# ============================================================
# SNAPSHOT DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("2. SNAPSHOTS PER CUSTOMER")
print("=" * 80)

snapshots = (
    df.groupBy("customer_unique_id")
    .count()
    .groupBy("count")
    .agg(
        F.count("*").alias("customers")
    )
    .orderBy("count")
)

snapshots.show(50, truncate=False)


# ============================================================
# ORDER COUNT VS CHURN
# ============================================================

print("\n" + "=" * 80)
print("3. CHURN RATE BY ORDER COUNT")
print("=" * 80)

order_churn = (
    df.groupBy("order_count_180d")
    .agg(
        F.count("*").alias("rows"),
        F.sum("churn_90d").alias("churned"),
        F.avg("churn_90d").alias("churn_rate")
    )
    .withColumn(
        "churn_rate_pct",
        F.round(F.col("churn_rate") * 100, 3)
    )
    .orderBy("order_count_180d")
)

order_churn.show(50, truncate=False)


# ============================================================
# RECENCY VS CHURN
# ============================================================

print("\n" + "=" * 80)
print("4. CHURN RATE BY RECENCY")
print("=" * 80)

recency_bands = (
    df.withColumn(
        "recency_band",
        F.when(F.col("recency_days") <= 30, "0-30")
        .when(F.col("recency_days") <= 60, "31-60")
        .when(F.col("recency_days") <= 90, "61-90")
        .when(F.col("recency_days") <= 120, "91-120")
        .when(F.col("recency_days") <= 150, "121-150")
        .otherwise("151-180")
    )
)

recency_churn = (
    recency_bands
    .groupBy("recency_band")
    .agg(
        F.count("*").alias("rows"),
        F.sum("churn_90d").alias("churned"),
        F.avg("churn_90d").alias("churn_rate")
    )
    .withColumn(
        "churn_rate_pct",
        F.round(F.col("churn_rate") * 100, 3)
    )
    .orderBy("recency_band")
)

recency_churn.show(truncate=False)


# ============================================================
# MONTHLY TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("5. CHURN RATE BY OBSERVATION MONTH")
print("=" * 80)

monthly = (
    df.groupBy("observation_date")
    .agg(
        F.count("*").alias("rows"),
        F.sum("churn_90d").alias("churned"),
        F.avg("churn_90d").alias("churn_rate"),
        F.countDistinct("customer_unique_id").alias("customers")
    )
    .withColumn(
        "churn_rate_pct",
        F.round(F.col("churn_rate") * 100, 3)
    )
    .orderBy("observation_date")
)

monthly.show(100, truncate=False)


# ============================================================
# CUSTOMER LIFETIME
# ============================================================

print("\n" + "=" * 80)
print("6. CUSTOMER LIFETIME DISTRIBUTION")
print("=" * 80)

df.select(
    "customer_lifetime_days"
).summary(
    "count",
    "mean",
    "stddev",
    "min",
    "25%",
    "50%",
    "75%",
    "max"
).show(truncate=False)


# ============================================================
# PURCHASE FREQUENCY
# ============================================================

print("\n" + "=" * 80)
print("7. PURCHASE FREQUENCY")
print("=" * 80)

df.select(
    "order_count_180d"
).summary(
    "count",
    "mean",
    "stddev",
    "min",
    "25%",
    "50%",
    "75%",
    "max"
).show(truncate=False)


# ============================================================
# MULTI-ORDER CUSTOMERS
# ============================================================

print("\n" + "=" * 80)
print("8. MULTI-ORDER OBSERVATIONS")
print("=" * 80)

multi_order = (
    df.withColumn(
        "customer_type",
        F.when(
            F.col("order_count_180d") == 1,
            "single_order"
        )
        .when(
            F.col("order_count_180d") >= 2,
            "repeat_order"
        )
        .otherwise("no_order")
    )
    .groupBy("customer_type")
    .agg(
        F.count("*").alias("rows"),
        F.countDistinct("customer_unique_id").alias("customers"),
        F.avg("churn_90d").alias("churn_rate")
    )
    .withColumn(
        "churn_rate_pct",
        F.round(F.col("churn_rate") * 100, 3)
    )
)

multi_order.show(truncate=False)


# ============================================================
# FEATURE SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("9. CORE FEATURE SUMMARY")
print("=" * 80)

core_features = [
    "order_count_180d",
    "revenue_180d",
    "payment_value_180d",
    "freight_value_180d",
    "item_count_180d",
    "unique_products_180d",
    "unique_categories_180d",
    "unique_sellers_180d",
    "recency_days",
    "customer_lifetime_days",
    "avg_order_value_180d",
    "orders_per_month_180d",
    "revenue_per_month_180d",
]

df.select(core_features).summary(
    "count",
    "mean",
    "stddev",
    "min",
    "25%",
    "50%",
    "75%",
    "max"
).show(truncate=False)


# ============================================================
# DATA QUALITY
# ============================================================

print("\n" + "=" * 80)
print("10. DATA QUALITY")
print("=" * 80)

for column in core_features + ["churn_90d"]:
    null_count = df.filter(
        F.col(column).isNull()
    ).count()

    zero_count = df.filter(
        F.col(column) == 0
    ).count()

    print(
        f"{column:35s} "
        f"nulls={null_count:8d} "
        f"zeros={zero_count:8d}"
    )


# ============================================================
# DISTINCT CUSTOMER FEATURE CHECK
# ============================================================

print("\n" + "=" * 80)
print("11. DISTINCT PRODUCT / CATEGORY / SELLER CHECK")
print("=" * 80)

df.select(
    "unique_products_180d",
    "unique_categories_180d",
    "unique_sellers_180d"
).summary(
    "count",
    "mean",
    "stddev",
    "min",
    "25%",
    "50%",
    "75%",
    "max"
).show(truncate=False)


# ============================================================
# HIGH-RISK STRUCTURE CHECK
# ============================================================

print("\n" + "=" * 80)
print("12. HIGH-RISK STRUCTURE CHECK")
print("=" * 80)

high_risk = (
    df.groupBy(
        "order_count_180d",
        "recency_days"
    )
    .agg(
        F.count("*").alias("rows"),
        F.avg("churn_90d").alias("churn_rate")
    )
    .withColumn(
        "churn_rate_pct",
        F.round(F.col("churn_rate") * 100, 3)
    )
    .orderBy(
        F.desc("rows")
    )
)

high_risk.show(50, truncate=False)


# ============================================================
# SAMPLE RECORDS
# ============================================================

print("\n" + "=" * 80)
print("13. SAMPLE V3 RECORDS")
print("=" * 80)

df.select(
    "customer_unique_id",
    "observation_date",
    "order_count_180d",
    "revenue_180d",
    "payment_value_180d",
    "item_count_180d",
    "unique_products_180d",
    "unique_categories_180d",
    "unique_sellers_180d",
    "recency_days",
    "customer_lifetime_days",
    "churn_90d"
).orderBy(
    "observation_date"
).show(20, truncate=False)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("V3 STRUCTURAL AUDIT COMPLETE")
print("=" * 80)

spark.stop()