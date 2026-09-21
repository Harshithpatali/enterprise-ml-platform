from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ============================================================
# CONFIG
# ============================================================

BASE = "hdfs:///data/silver/olist"
OUTPUT = "hdfs:///data/gold/olist/customer_churn_features_v3"

HISTORY_DAYS = 180
PREDICTION_DAYS = 90


# ============================================================
# SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("OlistCustomerChurnFeaturesV3")
    .master("local[*]")
    .config("spark.sql.ansi.enabled", "true")
    .config("spark.sql.shuffle.partitions", "16")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# LOAD SILVER DATA
# ============================================================

print("=" * 70)
print("LOADING SILVER DATA")
print("=" * 70)

orders = spark.read.parquet(f"{BASE}/orders")
customers = spark.read.parquet(f"{BASE}/customers")
items = spark.read.parquet(f"{BASE}/order_items")
payments = spark.read.parquet(f"{BASE}/payments")
reviews = spark.read.parquet(f"{BASE}/reviews")
products = spark.read.parquet(f"{BASE}/products")
categories = spark.read.parquet(f"{BASE}/category_translation")


# ============================================================
# COMPLETED ORDERS
# ============================================================

orders = (
    orders
    .filter(F.col("order_status") == "delivered")
    .filter(F.col("order_purchase_timestamp").isNotNull())
    .select(
        "order_id",
        "customer_id",
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    )
)

print(f"Completed orders: {orders.count()}")


# ============================================================
# CUSTOMER ID MAPPING
# ============================================================

customer_map = (
    customers
    .select(
        "customer_id",
        "customer_unique_id"
    )
    .dropDuplicates(["customer_id"])
)


orders = (
    orders
    .join(
        customer_map,
        on="customer_id",
        how="inner"
    )
    .select(
        "order_id",
        "customer_unique_id",
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    )
)

print(
    f"Orders with customer_unique_id: "
    f"{orders.count()}"
)


# ============================================================
# DATASET BOUNDARY
# ============================================================

min_order_date = orders.select(
    F.min("order_purchase_timestamp")
).first()[0]

max_order_date = orders.select(
    F.max("order_purchase_timestamp")
).first()[0]

print("\nOrder date range:")
print(f"Minimum: {min_order_date}")
print(f"Maximum: {max_order_date}")


# Last valid observation must have a complete
# 90-day future window.
max_valid_observation = (
    F.lit(max_order_date)
    .cast("timestamp")
    - F.expr(f"INTERVAL {PREDICTION_DAYS} DAYS")
)

max_valid_date = (
    orders
    .select(
        F.date_sub(
            F.to_date(F.lit(max_order_date)),
            PREDICTION_DAYS
        ).alias("max_valid_date")
    )
    .first()["max_valid_date"]
)

print(f"Maximum valid observation date: {max_valid_date}")


# ============================================================
# CUSTOMER ELIGIBILITY
# ============================================================
#
# A customer becomes eligible once they have completed
# at least one purchase.
#
# We then generate MONTHLY observations from their first
# completed purchase month through the valid dataset boundary.
# ============================================================

first_purchase = (
    orders
    .groupBy("customer_unique_id")
    .agg(
        F.min(
            "order_purchase_timestamp"
        ).alias("first_purchase_timestamp")
    )
    .withColumn(
        "first_observation_month",
        F.trunc(
            F.to_date("first_purchase_timestamp"),
            "month"
        )
    )
)


# ============================================================
# MONTHLY OBSERVATION CALENDAR
# ============================================================

print("\n" + "=" * 70)
print("BUILDING MONTHLY CUSTOMER OBSERVATIONS")
print("=" * 70)

calendar = (
    spark.range(
        1
    )
    .select(
        F.explode(
            F.sequence(
                F.to_date(
                    F.lit(
                        min_order_date
                    )
                ),
                F.to_date(
                    F.lit(
                        max_valid_date
                    )
                ),
                F.expr("INTERVAL 1 MONTH")
            )
        ).alias("observation_month")
    )
)

# Normalize to month end
calendar = calendar.withColumn(
    "observation_date",
    F.last_day("observation_month")
).select(
    "observation_date"
)

print("Observation calendar:")
calendar.show(20, truncate=False)


# ============================================================
# CUSTOMER × MONTH
# ============================================================

customer_months = (
    first_purchase
    .crossJoin(calendar)
    .filter(
        F.col("observation_date")
        >= F.last_day(
            F.col("first_observation_month")
        )
    )
    .filter(
        F.col("observation_date")
        <= F.lit(max_valid_date)
    )
    .select(
        "customer_unique_id",
        "observation_date"
    )
)

print(
    f"Customer-month rows before feature filtering: "
    f"{customer_months.count()}"
)


# ============================================================
# FEATURE WINDOW
# ============================================================

print("\n" + "=" * 70)
print("BUILDING 180-DAY FEATURES")
print("=" * 70)

history = (
    customer_months.alias("cm")
    .join(
        orders.alias("o"),
        (
            (F.col("cm.customer_unique_id")
             == F.col("o.customer_unique_id"))
            &
            (
                F.col("o.order_purchase_timestamp")
                > F.col("cm.observation_date")
                - F.expr(
                    f"INTERVAL {HISTORY_DAYS} DAYS"
                )
            )
            &
            (
                F.col("o.order_purchase_timestamp")
                <= F.col("cm.observation_date")
            )
        ),
        "left"
    )
)


# ============================================================
# ORDER-LEVEL ITEM FEATURES
# ============================================================

item_features = (
    items
    .join(
        products.select(
            "product_id",
            "product_category_name"
        ),
        "product_id",
        "left"
    )
    .join(
        categories,
        "product_category_name",
        "left"
    )
    .withColumn(
        "category_name",
        F.coalesce(
            F.col("product_category_name_english"),
            F.col("product_category_name")
        )
    )
    .groupBy("order_id")
    .agg(
        F.count("*").alias("item_count"),
        F.sum("price").alias("item_revenue"),
        F.sum("freight_value").alias("item_freight"),
        F.countDistinct(
            "product_id"
        ).alias("unique_products_order"),
        F.countDistinct(
            "seller_id"
        ).alias("unique_sellers_order"),
        F.countDistinct(
            "category_name"
        ).alias("unique_categories_order"),
    )
)


# ============================================================
# PAYMENT FEATURES
# ============================================================

payment_features = (
    payments
    .groupBy("order_id")
    .agg(
        F.sum("payment_value")
        .alias("payment_value"),

        F.avg("payment_installments")
        .alias("avg_installments"),

        F.countDistinct("payment_type")
        .alias("payment_type_count"),
    )
)


# ============================================================
# REVIEW FEATURES
# ============================================================

review_features = (
    reviews
    .groupBy("order_id")
    .agg(
        F.avg("review_score")
        .alias("review_score")
    )
)


# ============================================================
# ORDER FEATURE TABLE
# ============================================================

order_features = (
    orders
    .join(
        item_features,
        "order_id",
        "left"
    )
    .join(
        payment_features,
        "order_id",
        "left"
    )
    .join(
        review_features,
        "order_id",
        "left"
    )
    .withColumn(
        "delivery_delay_days",
        F.when(
            F.col("order_delivered_customer_date").isNotNull()
            &
            F.col("order_estimated_delivery_date").isNotNull(),
            F.datediff(
                F.to_date(
                    "order_delivered_customer_date"
                ),
                F.to_date(
                    "order_estimated_delivery_date"
                )
            )
        )
    )
)


# ============================================================
# AGGREGATE CUSTOMER-MONTH FEATURES
# ============================================================

features = (
    history
    .join(
        order_features.alias("of"),
        F.col("o.order_id")
        == F.col("of.order_id"),
        "left"
    )
    .groupBy(
        F.col("cm.customer_unique_id"),
        F.col("cm.observation_date")
    )
    .agg(

        # ----------------------------------------------------
        # Purchase frequency
        # ----------------------------------------------------

        F.countDistinct(
            "of.order_id"
        ).alias("order_count_180d"),

        # ----------------------------------------------------
        # Monetary
        # ----------------------------------------------------

        F.coalesce(
            F.sum("of.item_revenue"),
            F.lit(0.0)
        ).alias("revenue_180d"),

        F.coalesce(
            F.sum("of.payment_value"),
            F.lit(0.0)
        ).alias("payment_value_180d"),

        F.coalesce(
            F.sum("of.item_freight"),
            F.lit(0.0)
        ).alias("freight_value_180d"),

        # ----------------------------------------------------
        # Items
        # ----------------------------------------------------

        F.coalesce(
            F.sum("of.item_count"),
            F.lit(0)
        ).alias("item_count_180d"),

        # ----------------------------------------------------
        # Payment behavior
        # ----------------------------------------------------

        F.coalesce(
            F.avg("of.payment_type_count"),
            F.lit(0.0)
        ).alias("avg_payment_type_count_180d"),

        F.coalesce(
            F.avg("of.avg_installments"),
            F.lit(0.0)
        ).alias("avg_installments_180d"),

        # ----------------------------------------------------
        # Reviews
        # ----------------------------------------------------

        F.coalesce(
            F.avg("of.review_score"),
            F.lit(0.0)
        ).alias("avg_review_score_180d"),

        F.count(
            "of.review_score"
        ).alias("review_count_180d"),

        # ----------------------------------------------------
        # Delivery
        # ----------------------------------------------------

        F.coalesce(
            F.avg("of.delivery_delay_days"),
            F.lit(0.0)
        ).alias("avg_delivery_delay_180d"),

        F.coalesce(
            F.avg(
                F.when(
                    F.col("of.delivery_delay_days") > 0,
                    1.0
                ).otherwise(0.0)
            ),
            F.lit(0.0)
        ).alias("late_delivery_rate_180d"),

        # ----------------------------------------------------
        # Recency
        # ----------------------------------------------------

        F.max(
            "of.order_purchase_timestamp"
        ).alias("last_purchase_timestamp"),

        F.min(
            "of.order_purchase_timestamp"
        ).alias("first_purchase_timestamp_180d"),

        # ----------------------------------------------------
        # Customer-level diversity is calculated separately
        # from item-level data below.
        # ----------------------------------------------------
    )
)


# ============================================================
# CUSTOMER-LEVEL DISTINCT ITEM FEATURES
# ============================================================
#
# order_features is one row per order, so product/category/seller
# diversity must be calculated from item-level data to avoid
# double-counting entities appearing in multiple orders.
# ============================================================

item_details = (
    items
    .join(
        products.select(
            "product_id",
            "product_category_name"
        ),
        "product_id",
        "left"
    )
    .join(
        categories,
        "product_category_name",
        "left"
    )
    .withColumn(
        "category_name",
        F.coalesce(
            F.col("product_category_name_english"),
            F.col("product_category_name")
        )
    )
    .join(
        orders.select(
            "order_id",
            "customer_unique_id",
            "order_purchase_timestamp"
        ),
        "order_id",
        "inner"
    )
    .select(
        "order_id",
        "customer_unique_id",
        "order_purchase_timestamp",
        "product_id",
        "seller_id",
        "category_name"
    )
)

customer_month_item_features = (
    customer_months.alias("cm")
    .join(
        item_details.alias("i"),
        (
            F.col("cm.customer_unique_id")
            == F.col("i.customer_unique_id")
        )
        & (
            F.col("i.order_purchase_timestamp")
            > F.col("cm.observation_date")
            - F.expr(f"INTERVAL {HISTORY_DAYS} DAYS")
        )
        & (
            F.col("i.order_purchase_timestamp")
            <= F.col("cm.observation_date")
        ),
        "left"
    )
    .groupBy(
        F.col("cm.customer_unique_id"),
        F.col("cm.observation_date")
    )
    .agg(
        F.countDistinct("i.product_id")
        .alias("unique_products_180d"),

        F.countDistinct("i.category_name")
        .alias("unique_categories_180d"),

        F.countDistinct("i.seller_id")
        .alias("unique_sellers_180d")
    )
)

print("\nCustomer-month distinct item features created.")


# ============================================================
# JOIN CUSTOMER-LEVEL DISTINCT FEATURES
# ============================================================

features = (
    features
    .join(
        customer_month_item_features,
        ["customer_unique_id", "observation_date"],
        "left"
    )
    .fillna(
        {
            "unique_products_180d": 0,
            "unique_categories_180d": 0,
            "unique_sellers_180d": 0
        }
    )
)


# ============================================================
# ADD DERIVED FEATURES
# ============================================================

features = (
    features

    .withColumn(
        "recency_days",
        F.when(
            F.col("last_purchase_timestamp").isNotNull(),
            F.datediff(
                F.col("observation_date"),
                F.to_date(
                    "last_purchase_timestamp"
                )
            )
        ).otherwise(0)
    )

    .withColumn(
        "customer_lifetime_days",
        F.when(
            F.col("first_purchase_timestamp_180d").isNotNull(),
            F.datediff(
                F.col("observation_date"),
                F.to_date(
                    "first_purchase_timestamp_180d"
                )
            )
        ).otherwise(0)
    )

    .withColumn(
        "avg_order_value_180d",
        F.when(
            F.col("order_count_180d") > 0,
            F.col("revenue_180d")
            / F.col("order_count_180d")
        ).otherwise(0.0)
    )

    .withColumn(
        "orders_per_month_180d",
        F.col("order_count_180d")
        / F.lit(6.0)
    )

    .withColumn(
        "revenue_per_month_180d",
        F.col("revenue_180d")
        / F.lit(6.0)
    )
)


# ============================================================
# FUTURE 90-DAY LABEL
# ============================================================

print("\n" + "=" * 70)
print("BUILDING 90-DAY FUTURE LABEL")
print("=" * 70)

future_orders = (
    customer_months.alias("cm")
    .join(
        orders.alias("o"),
        (
            (F.col("cm.customer_unique_id")
             == F.col("o.customer_unique_id"))
            &
            (
                F.col("o.order_purchase_timestamp")
                > F.col("cm.observation_date")
            )
            &
            (
                F.col("o.order_purchase_timestamp")
                <= F.col("cm.observation_date")
                + F.expr(
                    f"INTERVAL {PREDICTION_DAYS} DAYS"
                )
            )
        ),
        "left"
    )
    .groupBy(
        F.col("cm.customer_unique_id"),
        F.col("cm.observation_date")
    )
    .agg(
        F.countDistinct(
            "o.order_id"
        ).alias("future_orders_90d")
    )
)


# ============================================================
# JOIN LABEL
# ============================================================

final_df = (
    features
    .join(
        future_orders,
        [
            "customer_unique_id",
            "observation_date"
        ],
        "left"
    )
    .fillna(
        {
            "future_orders_90d": 0
        }
    )
    .withColumn(
        "churn_90d",
        F.when(
            F.col("future_orders_90d") == 0,
            1
        ).otherwise(0)
    )
)


# ============================================================
# ELIGIBILITY FILTER
# ============================================================
#
# Remove observations with no historical activity.
#
# This changes the problem from:
# "Will a customer who has never purchased recently purchase?"
#
# to:
# "Among customers with recent historical activity,
#  who will fail to purchase again?"
# ============================================================

final_df = final_df.filter(
    F.col("order_count_180d") > 0
)


# ============================================================
# FINAL DATASET
# ============================================================

final_df = (
    final_df
    .withColumn(
        "observation_year",
        F.year("observation_date")
    )
    .withColumn(
        "observation_month",
        F.month("observation_date")
    )
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("V3 VALIDATION")
print("=" * 70)

print(
    f"Total rows: {final_df.count()}"
)

print(
    f"Unique customers: "
    f"{final_df.select('customer_unique_id').distinct().count()}"
)

print("\nTarget distribution:")

final_df.groupBy(
    "churn_90d"
).count().orderBy(
    "churn_90d"
).show()


print("\nObservation date range:")

final_df.select(
    F.min("observation_date").alias("min_date"),
    F.max("observation_date").alias("max_date")
).show()


print("\nRows by observation month:")

final_df.groupBy(
    "observation_date"
).count().orderBy(
    "observation_date"
).show(50)


print("\nSnapshots per customer:")

final_df.groupBy(
    "customer_unique_id"
).count().groupBy(
    "count"
).count().orderBy(
    "count"
).show(30)


# ============================================================
# WRITE GOLD V3
# ============================================================

print("\n" + "=" * 70)
print("WRITING GOLD V3")
print("=" * 70)

(
    final_df
    .repartition(
        "observation_year"
    )
    .write
    .mode("overwrite")
    .partitionBy(
        "observation_year"
    )
    .parquet(OUTPUT)
)


print("\n" + "=" * 70)
print("GOLD V3 CREATED")
print("=" * 70)

print(f"Location: {OUTPUT}")


spark.stop()