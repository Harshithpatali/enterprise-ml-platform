from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# CONFIGURATION
# ============================================================

GOLD_PATH = "hdfs:///data/gold/olist/customer_churn_features_v2"


# ============================================================
# START SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("OlistChurnStructureAnalysis")
    .master("local[*]")
    .config("spark.sql.ansi.enabled", "true")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# LOAD GOLD DATA
# ============================================================

print("=" * 70)
print("LOADING V2 GOLD DATA")
print("=" * 70)

df = spark.read.parquet(GOLD_PATH)

print(f"Total rows: {df.count()}")

print("\nColumns:")
print(df.columns)


# ============================================================
# 1. UNIQUE CUSTOMER STRUCTURE
# ============================================================

print("\n" + "=" * 70)
print("1. UNIQUE CUSTOMER STRUCTURE")
print("=" * 70)

customer_history = (
    df.groupBy("customer_unique_id")
    .agg(
        F.max("order_count_180d").alias("max_orders_180d"),
        F.max("customer_lifetime_days").alias(
            "lifetime_days"
        ),
        F.count("*").alias("snapshot_count"),
    )
)

total_customers = customer_history.count()

print(f"Unique customers: {total_customers}")


# ============================================================
# 2. SNAPSHOT COUNT DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("2. SNAPSHOTS PER CUSTOMER")
print("=" * 70)

snapshot_distribution = (
    customer_history
    .groupBy("snapshot_count")
    .count()
    .orderBy("snapshot_count")
)

snapshot_distribution.show(
    30,
    truncate=False
)


# ============================================================
# 3. CUSTOMERS WITH 1 VS 2+ HISTORICAL SNAPSHOTS
# ============================================================

print("\n" + "=" * 70)
print("3. CUSTOMER REPEAT HISTORY")
print("=" * 70)

repeat_summary = (
    customer_history
    .withColumn(
        "customer_type",
        F.when(
            F.col("snapshot_count") == 1,
            "one_snapshot"
        )
        .otherwise("multiple_snapshots")
    )
    .groupBy("customer_type")
    .count()
)

repeat_summary.show()


# ============================================================
# 4. ORDER COUNT DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("4. ORDER COUNT IN 180-DAY WINDOW")
print("=" * 70)

order_distribution = (
    df.groupBy("order_count_180d")
    .agg(
        F.count("*").alias("rows"),
        F.avg("churn_90d").alias("churn_rate"),
        F.avg("revenue_180d").alias("avg_revenue"),
        F.avg("recency_days").alias("avg_recency"),
    )
    .orderBy("order_count_180d")
)

order_distribution.show(
    30,
    truncate=False
)


# ============================================================
# 5. CHURN BY ORDER COUNT
# ============================================================

print("\n" + "=" * 70)
print("5. CHURN RATE BY ORDER COUNT")
print("=" * 70)

churn_by_orders = (
    df.groupBy("order_count_180d")
    .agg(
        F.count("*").alias("customers"),
        F.sum(
            F.when(
                F.col("churn_90d") == 1,
                1
            ).otherwise(0)
        ).alias("churned"),
        F.sum(
            F.when(
                F.col("churn_90d") == 0,
                1
            ).otherwise(0)
        ).alias("retained"),
        F.round(
            F.avg("churn_90d") * 100,
            3
        ).alias("churn_rate_pct"),
    )
    .orderBy("order_count_180d")
)

churn_by_orders.show(
    30,
    truncate=False
)


# ============================================================
# 6. BIN ORDER COUNT
# ============================================================

print("\n" + "=" * 70)
print("6. ORDER COUNT BANDS")
print("=" * 70)

order_bands = (
    df.withColumn(
        "order_count_band",
        F.when(
            F.col("order_count_180d") == 0,
            "0"
        )
        .when(
            F.col("order_count_180d") == 1,
            "1"
        )
        .when(
            F.col("order_count_180d") == 2,
            "2"
        )
        .when(
            F.col("order_count_180d") == 3,
            "3"
        )
        .otherwise("4+")
    )
    .groupBy("order_count_band")
    .agg(
        F.count("*").alias("customers"),
        F.sum(
            F.when(
                F.col("churn_90d") == 1,
                1
            ).otherwise(0)
        ).alias("churned"),
        F.sum(
            F.when(
                F.col("churn_90d") == 0,
                1
            ).otherwise(0)
        ).alias("retained"),
        F.round(
            F.avg("churn_90d") * 100,
            3
        ).alias("churn_rate_pct"),
    )
)

order_bands.show(
    truncate=False
)


# ============================================================
# 7. RECENCY ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("7. CHURN BY RECENCY")
print("=" * 70)

recency_bands = (
    df.withColumn(
        "recency_band",
        F.when(
            F.col("recency_days") <= 30,
            "0-30"
        )
        .when(
            F.col("recency_days") <= 60,
            "31-60"
        )
        .when(
            F.col("recency_days") <= 90,
            "61-90"
        )
        .when(
            F.col("recency_days") <= 120,
            "91-120"
        )
        .otherwise("120+")
    )
    .groupBy("recency_band")
    .agg(
        F.count("*").alias("customers"),
        F.round(
            F.avg("churn_90d") * 100,
            3
        ).alias("churn_rate_pct"),
    )
    .orderBy("recency_band")
)

recency_bands.show(
    truncate=False
)


# ============================================================
# 8. TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("8. OVERALL TARGET DISTRIBUTION")
print("=" * 70)

target_summary = (
    df.groupBy("churn_90d")
    .count()
    .withColumn(
        "percentage",
        F.round(
            F.col("count") /
            F.sum("count").over(
                __import__(
                    "pyspark"
                ).sql.Window.partitionBy()
            ) * 100,
            3
        )
    )
)

target_summary.show()


# ============================================================
# 9. KEY FEATURE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("9. KEY FEATURE SUMMARY")
print("=" * 70)

df.select(
    "order_count_180d",
    "item_count_180d",
    "unique_products_180d",
    "unique_categories_180d",
    "unique_sellers_180d",
    "payment_value_180d",
    "revenue_180d",
    "recency_days",
    "customer_lifetime_days",
    "churn_90d",
).describe().show()


# ============================================================
# 10. CUSTOMER-LEVEL HISTORY ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("10. CUSTOMER HISTORY")
print("=" * 70)

customer_history.groupBy(
    "max_orders_180d"
).agg(
    F.count("*").alias("customers")
).orderBy(
    "max_orders_180d"
).show(
    30,
    truncate=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("STRUCTURAL ANALYSIS COMPLETE")
print("=" * 70)

spark.stop()