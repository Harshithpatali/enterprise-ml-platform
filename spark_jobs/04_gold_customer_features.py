from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    lit,
    when,
    count,
    sum,
    avg,
    max,
    min,
    countDistinct,
    datediff,
    date_trunc,
    last_day,
    to_date,
    expr,
    coalesce,
    row_number,
)
from pyspark.sql.window import Window


BRONZE_PATH = "hdfs:///data/bronze/olist"
SILVER_PATH = "hdfs:///data/silver/olist"
GOLD_PATH = "hdfs:///data/gold/olist/customer_churn_features"


def create_spark():
    return (
        SparkSession.builder
        .appName("OlistGoldCustomerChurn")
        .config("spark.sql.ansi.enabled", "true")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def read_silver(spark):
    print("\nReading Silver datasets...")

    customers = spark.read.parquet(f"{SILVER_PATH}/customers")
    orders = spark.read.parquet(f"{SILVER_PATH}/orders")
    order_items = spark.read.parquet(f"{SILVER_PATH}/order_items")
    payments = spark.read.parquet(f"{SILVER_PATH}/payments")
    reviews = spark.read.parquet(f"{SILVER_PATH}/reviews")
    products = spark.read.parquet(f"{SILVER_PATH}/products")
    category_translation = spark.read.parquet(
        f"{SILVER_PATH}/category_translation"
    )

    return (
        customers,
        orders,
        order_items,
        payments,
        reviews,
        products,
        category_translation,
    )


def prepare_customers(customers):
    """
    Customer dimension.

    customer_unique_id is used as the analytical customer identifier.
    """

    return (
        customers
        .select(
            "customer_id",
            "customer_unique_id",
            "customer_city",
            "customer_state",
        )
        .dropDuplicates(["customer_id"])
    )


def prepare_orders(orders, customers):
    """
    Create the core order-level analytical table.

    Only delivered orders are considered completed purchases.
    """

    completed_orders = (
        orders
        .filter(col("order_status") == "delivered")
        .filter(col("order_purchase_timestamp").isNotNull())
        .join(
            customers,
            on="customer_id",
            how="inner",
        )
        .select(
            "order_id",
            "customer_id",
            "customer_unique_id",
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        )
    )

    completed_orders = (
        completed_orders
        .withColumn(
            "purchase_date",
            to_date("order_purchase_timestamp"),
        )
        .withColumn(
            "observation_month",
            last_day(
                date_trunc(
                    "month",
                    col("order_purchase_timestamp"),
                )
            ),
        )
        .withColumn(
            "delivery_delay_days",
            when(
                col("order_delivered_customer_date").isNotNull()
                & col("order_estimated_delivery_date").isNotNull(),
                datediff(
                    col("order_delivered_customer_date"),
                    col("order_estimated_delivery_date"),
                ),
            ).otherwise(lit(0)),
        )
        .withColumn(
            "is_late",
            when(col("delivery_delay_days") > 0, 1).otherwise(0),
        )
    )

    return completed_orders


def prepare_item_features(order_items, products, category_translation):
    """
    Aggregate order-item information to order level first.

    This avoids double counting when joining with payments/reviews.
    """

    items = (
        order_items
        .join(
            products.select(
                "product_id",
                "product_category_name",
            ),
            on="product_id",
            how="left",
        )
        .join(
            category_translation,
            on="product_category_name",
            how="left",
        )
    )

    item_features = (
        items
        .groupBy("order_id")
        .agg(
            count("*").alias("item_count"),
            sum("price").alias("item_revenue"),
            sum("freight_value").alias("freight_value"),
            countDistinct("product_id").alias("unique_products"),
            countDistinct("seller_id").alias("unique_sellers"),
            countDistinct(
                coalesce(
                    col("product_category_name_english"),
                    col("product_category_name"),
                )
            ).alias("unique_categories"),
        )
    )

    return item_features


def prepare_payment_features(payments):
    """
    Aggregate payment information to order level.
    """

    return (
        payments
        .groupBy("order_id")
        .agg(
            sum("payment_value").alias("payment_value"),
            avg("payment_installments").alias(
                "avg_payment_installments"
            ),
            countDistinct("payment_type").alias(
                "payment_type_count"
            ),
        )
    )


def prepare_review_features(reviews):
    """
    Aggregate review information to order level.
    """

    return (
        reviews
        .groupBy("order_id")
        .agg(
            avg("review_score").alias("avg_review_score"),
            count("*").alias("review_count"),
        )
    )


def build_order_level_table(
    orders,
    item_features,
    payment_features,
    review_features,
):
    """
    Combine all order-level features.

    Every fact table has already been aggregated to order level,
    preventing revenue/payment/review multiplication.
    """

    order_level = (
        orders
        .join(
            item_features,
            on="order_id",
            how="left",
        )
        .join(
            payment_features,
            on="order_id",
            how="left",
        )
        .join(
            review_features,
            on="order_id",
            how="left",
        )
    )

    numeric_columns = [
        "item_count",
        "item_revenue",
        "freight_value",
        "unique_products",
        "unique_sellers",
        "unique_categories",
        "payment_value",
        "avg_payment_installments",
        "payment_type_count",
        "avg_review_score",
        "review_count",
    ]

    for c in numeric_columns:
        order_level = order_level.withColumn(
            c,
            coalesce(col(c), lit(0)),
        )

    return order_level


def create_observation_dates(order_level):
    """
    Generate customer-month observation snapshots.

    A snapshot exists when the customer had at least one
    completed purchase during that month.

    Observation date = final calendar day of the month.
    """

    observations = (
        order_level
        .select(
            "customer_unique_id",
            "observation_month",
        )
        .dropDuplicates()
        .withColumnRenamed(
            "observation_month",
            "observation_date",
        )
    )

    return observations


def build_features(order_level, observations):
    """
    Build point-in-time features using the previous 180 days.

    IMPORTANT:
    Only information available on or before observation_date
    is used.
    """

    orders_for_features = order_level.select(
        "order_id",
        "customer_unique_id",
        "order_purchase_timestamp",
        "item_count",
        "item_revenue",
        "freight_value",
        "unique_products",
        "unique_sellers",
        "unique_categories",
        "payment_value",
        "avg_payment_installments",
        "payment_type_count",
        "avg_review_score",
        "review_count",
        "delivery_delay_days",
        "is_late",
    )

    joined = (
        observations.alias("obs")
        .join(
            orders_for_features.alias("ord"),
            (
                (col("obs.customer_unique_id")
                 == col("ord.customer_unique_id"))
                &
                (
                    col("ord.order_purchase_timestamp")
                    >= expr(
                        "obs.observation_date - INTERVAL 180 DAYS"
                    )
                )
                &
                (
                    col("ord.order_purchase_timestamp")
                    <= expr(
                        "CAST(obs.observation_date AS TIMESTAMP)"
                    )
                )
            ),
            how="left",
        )
    )

    features = (
        joined
        .groupBy(
            col("obs.customer_unique_id").alias(
                "customer_unique_id"
            ),
            col("obs.observation_date").alias(
                "observation_date"
            ),
        )
        .agg(
            countDistinct("ord.order_id").alias(
                "order_count_180d"
            ),

            sum("ord.item_revenue").alias(
                "revenue_180d"
            ),

            sum("ord.payment_value").alias(
                "payment_value_180d"
            ),

            sum("ord.freight_value").alias(
                "freight_value_180d"
            ),

            sum("ord.item_count").alias(
                "item_count_180d"
            ),

            countDistinct("ord.unique_products").alias(
                "distinct_product_baskets_180d"
            ),

            avg("ord.unique_products").alias(
                "avg_products_per_order_180d"
            ),

            avg("ord.unique_categories").alias(
                "avg_categories_per_order_180d"
            ),

            avg("ord.unique_sellers").alias(
                "avg_sellers_per_order_180d"
            ),

            avg("ord.payment_type_count").alias(
                "avg_payment_type_count_180d"
            ),

            avg("ord.avg_payment_installments").alias(
                "avg_installments_180d"
            ),

            avg("ord.avg_review_score").alias(
                "avg_review_score_180d"
            ),

            sum("ord.review_count").alias(
                "review_count_180d"
            ),

            avg("ord.delivery_delay_days").alias(
                "avg_delivery_delay_180d"
            ),

            avg("ord.is_late").alias(
                "late_delivery_rate_180d"
            ),

            max("ord.order_purchase_timestamp").alias(
                "last_purchase_timestamp"
            ),

            min("ord.order_purchase_timestamp").alias(
                "first_purchase_timestamp_180d"
            ),
        )
    )

    features = (
        features
        .withColumn(
            "recency_days",
            datediff(
                col("observation_date"),
                to_date("last_purchase_timestamp"),
            ),
        )
        .withColumn(
            "customer_lifetime_days",
            datediff(
                col("observation_date"),
                to_date("first_purchase_timestamp_180d"),
            ),
        )
        .withColumn(
            "avg_order_value_180d",
            when(
                col("order_count_180d") > 0,
                col("revenue_180d")
                / col("order_count_180d"),
            ).otherwise(0),
        )
        .withColumn(
            "orders_per_month_180d",
            col("order_count_180d") / lit(6.0),
        )
        .withColumn(
            "revenue_per_month_180d",
            col("revenue_180d") / lit(6.0),
        )
    )

    return features


def build_churn_label(order_level, features):
    """
    Churn definition:

    churn_90d = 1
        if the customer has NO completed purchase
        during the 90 days following observation_date.

    churn_90d = 0
        if the customer makes at least one completed
        purchase during that period.
    """

    future_orders = order_level.select(
        "customer_unique_id",
        "order_purchase_timestamp",
    )

    labeled = (
        features.alias("f")
        .join(
            future_orders.alias("future"),
            (
                (col("f.customer_unique_id")
                 == col("future.customer_unique_id"))
                &
                (
                    col("future.order_purchase_timestamp")
                    > expr(
                        "CAST(f.observation_date AS TIMESTAMP)"
                    )
                )
                &
                (
                    col("future.order_purchase_timestamp")
                    <= expr(
                        "CAST(f.observation_date AS TIMESTAMP)"
                        " + INTERVAL 90 DAYS"
                    )
                )
            ),
            how="left",
        )
        .groupBy(
            "f.customer_unique_id",
            "f.observation_date",
            *[
                f"f.{c}"
                for c in features.columns
                if c not in [
                    "customer_unique_id",
                    "observation_date",
                ]
            ],
        )
        .agg(
            count("future.order_purchase_timestamp").alias(
                "future_orders_90d"
            )
        )
        .withColumn(
            "churn_90d",
            when(col("future_orders_90d") == 0, 1)
            .otherwise(0),
        )
    )

    return labeled


def clean_gold_dataset(df):
    """
    Final feature cleanup.
    """

    numeric_columns = [
        "order_count_180d",
        "revenue_180d",
        "payment_value_180d",
        "freight_value_180d",
        "item_count_180d",
        "distinct_product_baskets_180d",
        "avg_products_per_order_180d",
        "avg_categories_per_order_180d",
        "avg_sellers_per_order_180d",
        "avg_payment_type_count_180d",
        "avg_installments_180d",
        "avg_review_score_180d",
        "review_count_180d",
        "avg_delivery_delay_180d",
        "late_delivery_rate_180d",
        "recency_days",
        "customer_lifetime_days",
        "avg_order_value_180d",
        "orders_per_month_180d",
        "revenue_per_month_180d",
        "future_orders_90d",
        "churn_90d",
    ]

    for c in numeric_columns:
        df = df.withColumn(
            c,
            coalesce(col(c), lit(0)),
        )

    df = (
        df
        .withColumn(
            "observation_year",
            expr("year(observation_date)"),
        )
        .withColumn(
            "observation_month",
            expr("month(observation_date)"),
        )
    )

    return df


def validate_gold(df):
    """
    Basic production-style validation.
    """

    print("\n========== GOLD DATASET VALIDATION ==========")

    total_rows = df.count()

    print(f"Total rows: {total_rows:,}")

    if total_rows == 0:
        raise ValueError(
            "Gold dataset contains zero rows."
        )

    print("\nSchema:")
    df.printSchema()

    print("\nChurn distribution:")

    (
        df.groupBy("churn_90d")
        .count()
        .orderBy("churn_90d")
        .show()
    )

    print("\nObservation date range:")

    (
        df.select(
            min("observation_date").alias("min_date"),
            max("observation_date").alias("max_date"),
        )
        .show()
    )

    print("\nCustomer count:")

    (
        df.select("customer_unique_id")
        .distinct()
        .count()
    )

    print("\nFeature summary:")

    (
        df.select(
            "order_count_180d",
            "revenue_180d",
            "recency_days",
            "avg_order_value_180d",
            "late_delivery_rate_180d",
        )
        .summary()
        .show()
    )

    print("\nSample rows:")

    (
        df.orderBy(
            col("observation_date").desc()
        )
        .show(10, truncate=False)
    )

    print("\n=============================================")


def main():

    spark = create_spark()

    spark.sparkContext.setLogLevel("WARN")

    print("\n=============================================")
    print(" OLIST GOLD CUSTOMER CHURN PIPELINE")
    print("=============================================")

    (
        customers,
        orders,
        order_items,
        payments,
        reviews,
        products,
        category_translation,
    ) = read_silver(spark)

    print("\nPreparing customers...")
    customers = prepare_customers(customers)

    print("Preparing completed orders...")
    orders = prepare_orders(
        orders,
        customers,
    )

    print("Preparing item-level features...")
    item_features = prepare_item_features(
        order_items,
        products,
        category_translation,
    )

    print("Preparing payment features...")
    payment_features = prepare_payment_features(
        payments
    )

    print("Preparing review features...")
    review_features = prepare_review_features(
        reviews
    )

    print("Building order-level analytical table...")
    order_level = build_order_level_table(
        orders,
        item_features,
        payment_features,
        review_features,
    )

    print("\nOrder-level records:")
    print(f"{order_level.count():,}")

    print("\nCreating customer observation dates...")
    observations = create_observation_dates(
        order_level
    )

    print(
        f"Observation rows: {observations.count():,}"
    )

    print("\nBuilding 180-day customer features...")
    features = build_features(
        order_level,
        observations,
    )

    print(
        f"Feature rows: {features.count():,}"
    )

    print("\nCreating 90-day churn labels...")
    gold = build_churn_label(
        order_level,
        features,
    )

    print("\nCleaning Gold dataset...")
    gold = clean_gold_dataset(gold)

    # Remove duplicate customer-observation rows.
    gold = gold.dropDuplicates(
        [
            "customer_unique_id",
            "observation_date",
        ]
    )

    validate_gold(gold)

    print("\nWriting Gold dataset...")

    (
        gold
        .repartition(
            "observation_year",
            "observation_month",
        )
        .write
        .mode("overwrite")
        .partitionBy(
            "observation_year",
            "observation_month",
        )
        .parquet(GOLD_PATH)
    )

    print("\n=============================================")
    print(" GOLD LAYER CREATED SUCCESSFULLY")
    print("=============================================")
    print(f"Location: {GOLD_PATH}")

    spark.stop()


if __name__ == "__main__":
    main()