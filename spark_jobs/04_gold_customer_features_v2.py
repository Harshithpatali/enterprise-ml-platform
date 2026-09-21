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
)


SILVER_PATH = "hdfs:///data/silver/olist"

GOLD_PATH = (
    "hdfs:///data/gold/olist/customer_churn_features_v2"
)


def create_spark():

    return (
        SparkSession.builder
        .appName("OlistGoldCustomerChurnV2")
        .config("spark.sql.ansi.enabled", "true")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


# ============================================================
# READ SILVER
# ============================================================

def read_silver(spark):

    print("\nReading Silver datasets...")

    customers = spark.read.parquet(
        f"{SILVER_PATH}/customers"
    )

    orders = spark.read.parquet(
        f"{SILVER_PATH}/orders"
    )

    order_items = spark.read.parquet(
        f"{SILVER_PATH}/order_items"
    )

    payments = spark.read.parquet(
        f"{SILVER_PATH}/payments"
    )

    reviews = spark.read.parquet(
        f"{SILVER_PATH}/reviews"
    )

    products = spark.read.parquet(
        f"{SILVER_PATH}/products"
    )

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


# ============================================================
# CUSTOMER DIMENSION
# ============================================================

def prepare_customers(customers):

    return (
        customers
        .select(
            "customer_id",
            "customer_unique_id",
            "customer_city",
            "customer_state",
        )
        .dropDuplicates(
            ["customer_id"]
        )
    )


# ============================================================
# COMPLETED ORDERS
# ============================================================

def prepare_orders(
    orders,
    customers,
):

    completed_orders = (
        orders
        .filter(
            col("order_status") == "delivered"
        )
        .filter(
            col("order_purchase_timestamp").isNotNull()
        )
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
            to_date(
                "order_purchase_timestamp"
            ),
        )

        .withColumn(
            "observation_month",
            last_day(
                date_trunc(
                    "month",
                    col(
                        "order_purchase_timestamp"
                    ),
                )
            ),
        )

        .withColumn(
            "delivery_delay_days",
            when_delivery_delay(),
        )

        .withColumn(
            "is_late",
            when(
                col(
                    "delivery_delay_days"
                ) > 0,
                1,
            ).otherwise(0),
        )
    )

    return completed_orders


def when_delivery_delay():

    return when(
        col(
            "order_delivered_customer_date"
        ).isNotNull()
        &
        col(
            "order_estimated_delivery_date"
        ).isNotNull(),

        datediff(
            col(
                "order_delivered_customer_date"
            ),
            col(
                "order_estimated_delivery_date"
            ),
        ),
    ).otherwise(
        lit(0)
    )


# ============================================================
# ITEM-LEVEL DATA
# ============================================================

def prepare_items(
    order_items,
    products,
    category_translation,
):

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

    items = (
        items
        .withColumn(
            "category_name",
            coalesce(
                col(
                    "product_category_name_english"
                ),
                col(
                    "product_category_name"
                ),
            ),
        )
    )

    return items


# ============================================================
# ORDER-LEVEL ITEM FEATURES
# ============================================================

def prepare_item_order_features(
    items,
):

    return (
        items
        .groupBy(
            "order_id"
        )
        .agg(

            count("*").alias(
                "item_count"
            ),

            sum("price").alias(
                "item_revenue"
            ),

            sum("freight_value").alias(
                "freight_value"
            ),

            countDistinct(
                "product_id"
            ).alias(
                "unique_products_order"
            ),

            countDistinct(
                "seller_id"
            ).alias(
                "unique_sellers_order"
            ),

            countDistinct(
                "category_name"
            ).alias(
                "unique_categories_order"
            ),
        )
    )


# ============================================================
# PAYMENT FEATURES
# ============================================================

def prepare_payment_features(
    payments,
):

    return (
        payments
        .groupBy(
            "order_id"
        )
        .agg(

            sum(
                "payment_value"
            ).alias(
                "payment_value"
            ),

            avg(
                "payment_installments"
            ).alias(
                "avg_payment_installments"
            ),

            countDistinct(
                "payment_type"
            ).alias(
                "payment_type_count"
            ),
        )
    )


# ============================================================
# REVIEW FEATURES
# ============================================================

def prepare_review_features(
    reviews,
):

    return (
        reviews
        .groupBy(
            "order_id"
        )
        .agg(

            avg(
                "review_score"
            ).alias(
                "avg_review_score"
            ),

            count("*").alias(
                "review_count"
            ),
        )
    )


# ============================================================
# ORDER-LEVEL ANALYTICAL TABLE
# ============================================================

def build_order_level_table(
    orders,
    item_features,
    payment_features,
    review_features,
):

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

        "unique_products_order",
        "unique_sellers_order",
        "unique_categories_order",

        "payment_value",
        "avg_payment_installments",
        "payment_type_count",

        "avg_review_score",
        "review_count",
    ]

    for c in numeric_columns:

        order_level = order_level.withColumn(
            c,
            coalesce(
                col(c),
                lit(0),
            ),
        )

    return order_level


# ============================================================
# CUSTOMER-MONTH OBSERVATIONS
# ============================================================

def create_observation_dates(
    order_level,
):

    return (
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


# ============================================================
# 180-DAY CUSTOMER FEATURES
# ============================================================

def build_features(
    order_level,
    observations,
    items,
):

    # --------------------------------------------------------
    # Basic order-level behavior
    # --------------------------------------------------------

    orders_for_features = (
        order_level
        .select(

            "order_id",
            "customer_unique_id",
            "order_purchase_timestamp",

            "item_count",
            "item_revenue",
            "freight_value",

            "payment_value",

            "avg_payment_installments",
            "payment_type_count",

            "avg_review_score",
            "review_count",

            "delivery_delay_days",
            "is_late",
        )
    )

    # --------------------------------------------------------
    # Actual product/category/seller information
    # --------------------------------------------------------

    item_customer_features = (
        items

        .join(
            order_level.select(
                "order_id",
                "customer_unique_id",
                "order_purchase_timestamp",
            ),
            on="order_id",
            how="inner",
        )
    )

    # --------------------------------------------------------
    # Join customer snapshots to historical orders
    # --------------------------------------------------------

    joined = (
        observations.alias("obs")

        .join(
            orders_for_features.alias("ord"),

            (
                (
                    col(
                        "obs.customer_unique_id"
                    )
                    ==
                    col(
                        "ord.customer_unique_id"
                    )
                )

                &

                (
                    col(
                        "ord.order_purchase_timestamp"
                    )
                    >=
                    expr(
                        "obs.observation_date "
                        "- INTERVAL 180 DAYS"
                    )
                )

                &

                (
                    col(
                        "ord.order_purchase_timestamp"
                    )
                    <=
                    expr(
                        "CAST("
                        "obs.observation_date "
                        "AS TIMESTAMP)"
                    )
                )
            ),

            how="left",
        )
    )

    # --------------------------------------------------------
    # Basic behavioral aggregation
    # --------------------------------------------------------

    features = (
        joined

        .groupBy(
            col(
                "obs.customer_unique_id"
            ).alias(
                "customer_unique_id"
            ),

            col(
                "obs.observation_date"
            ).alias(
                "observation_date"
            ),
        )

        .agg(

            countDistinct(
                "ord.order_id"
            ).alias(
                "order_count_180d"
            ),

            sum(
                "ord.item_revenue"
            ).alias(
                "revenue_180d"
            ),

            sum(
                "ord.payment_value"
            ).alias(
                "payment_value_180d"
            ),

            sum(
                "ord.freight_value"
            ).alias(
                "freight_value_180d"
            ),

            sum(
                "ord.item_count"
            ).alias(
                "item_count_180d"
            ),

            avg(
                "ord.payment_type_count"
            ).alias(
                "avg_payment_type_count_180d"
            ),

            avg(
                "ord.avg_payment_installments"
            ).alias(
                "avg_installments_180d"
            ),

            avg(
                "ord.avg_review_score"
            ).alias(
                "avg_review_score_180d"
            ),

            sum(
                "ord.review_count"
            ).alias(
                "review_count_180d"
            ),

            avg(
                "ord.delivery_delay_days"
            ).alias(
                "avg_delivery_delay_180d"
            ),

            avg(
                "ord.is_late"
            ).alias(
                "late_delivery_rate_180d"
            ),

            max(
                "ord.order_purchase_timestamp"
            ).alias(
                "last_purchase_timestamp"
            ),

            min(
                "ord.order_purchase_timestamp"
            ).alias(
                "first_purchase_timestamp_180d"
            ),
        )
    )

    # --------------------------------------------------------
    # ACTUAL DISTINCT PRODUCT/CATEGORY/SELLER FEATURES
    # --------------------------------------------------------

    item_joined = (
        observations.alias("obs")

        .join(
            item_customer_features.alias(
                "item"
            ),

            (
                (
                    col(
                        "obs.customer_unique_id"
                    )
                    ==
                    col(
                        "item.customer_unique_id"
                    )
                )

                &

                (
                    col(
                        "item.order_purchase_timestamp"
                    )
                    >=
                    expr(
                        "obs.observation_date "
                        "- INTERVAL 180 DAYS"
                    )
                )

                &

                (
                    col(
                        "item.order_purchase_timestamp"
                    )
                    <=
                    expr(
                        "CAST("
                        "obs.observation_date "
                        "AS TIMESTAMP)"
                    )
                )
            ),

            how="left",
        )
    )

    item_features = (
        item_joined

        .groupBy(
            col(
                "obs.customer_unique_id"
            ).alias(
                "customer_unique_id"
            ),

            col(
                "obs.observation_date"
            ).alias(
                "observation_date"
            ),
        )

        .agg(

            countDistinct(
                "item.product_id"
            ).alias(
                "unique_products_180d"
            ),

            countDistinct(
                "item.category_name"
            ).alias(
                "unique_categories_180d"
            ),

            countDistinct(
                "item.seller_id"
            ).alias(
                "unique_sellers_180d"
            ),
        )
    )

    # --------------------------------------------------------
    # Combine feature groups
    # --------------------------------------------------------

    features = (
        features

        .join(
            item_features,
            on=[
                "customer_unique_id",
                "observation_date",
            ],
            how="left",
        )
    )

    # --------------------------------------------------------
    # Derived behavioral features
    # --------------------------------------------------------

    features = (

        features

        .withColumn(
            "recency_days",
            datediff(
                col(
                    "observation_date"
                ),
                to_date(
                    col(
                        "last_purchase_timestamp"
                    )
                ),
            ),
        )

        .withColumn(
            "customer_lifetime_days",
            datediff(
                col(
                    "observation_date"
                ),
                to_date(
                    col(
                        "first_purchase_timestamp_180d"
                    )
                ),
            ),
        )

        .withColumn(
            "avg_order_value_180d",

            when(
                col(
                    "order_count_180d"
                ) > 0,

                col(
                    "revenue_180d"
                )
                /
                col(
                    "order_count_180d"
                ),
            )

            .otherwise(
                lit(0)
            ),
        )

        .withColumn(
            "orders_per_month_180d",

            col(
                "order_count_180d"
            )
            /
            lit(6.0),
        )

        .withColumn(
            "revenue_per_month_180d",

            col(
                "revenue_180d"
            )
            /
            lit(6.0),
        )
    )

    return features


# ============================================================
# CHURN LABEL
# ============================================================

def build_churn_label(
    order_level,
    features,
):

    future_orders = (
        order_level
        .select(
            "customer_unique_id",
            "order_purchase_timestamp",
        )
    )

    feature_columns = [
        c
        for c in features.columns
        if c not in [
            "customer_unique_id",
            "observation_date",
        ]
    ]

    labeled = (

        features.alias("f")

        .join(
            future_orders.alias("future"),

            (
                (
                    col(
                        "f.customer_unique_id"
                    )
                    ==
                    col(
                        "future.customer_unique_id"
                    )
                )

                &

                (
                    col(
                        "future.order_purchase_timestamp"
                    )
                    >
                    expr(
                        "CAST("
                        "f.observation_date "
                        "AS TIMESTAMP)"
                    )
                )

                &

                (
                    col(
                        "future.order_purchase_timestamp"
                    )
                    <=
                    expr(
                        "CAST("
                        "f.observation_date "
                        "AS TIMESTAMP)"
                        " + INTERVAL 90 DAYS"
                    )
                )
            ),

            how="left",
        )

        .groupBy(
            col(
                "f.customer_unique_id"
            ).alias(
                "customer_unique_id"
            ),

            col(
                "f.observation_date"
            ).alias(
                "observation_date"
            ),

            *[
                col(
                    f"f.{c}"
                )
                for c in feature_columns
            ],
        )

        .agg(
            count(
                "future.order_purchase_timestamp"
            ).alias(
                "future_orders_90d"
            )
        )

        .withColumn(
            "churn_90d",

            when(
                col(
                    "future_orders_90d"
                ) == 0,
                1,
            )

            .otherwise(
                0
            ),
        )
    )

    return labeled


# ============================================================
# CLEAN DATASET
# ============================================================

def clean_gold_dataset(
    df,
):

    numeric_columns = [

        "order_count_180d",
        "revenue_180d",
        "payment_value_180d",
        "freight_value_180d",
        "item_count_180d",

        "unique_products_180d",
        "unique_categories_180d",
        "unique_sellers_180d",

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
            coalesce(
                col(c),
                lit(0),
            ),
        )

    df = (
        df

        .withColumn(
            "observation_year",
            expr(
                "year(observation_date)"
            ),
        )

        .withColumn(
            "observation_month",
            expr(
                "month(observation_date)"
            ),
        )
    )

    return df


# ============================================================
# VALIDATION
# ============================================================

def validate_gold(
    df,
):

    print(
        "\n============================================"
    )

    print(
        " GOLD V2 VALIDATION"
    )

    print(
        "============================================"
    )

    total_rows = df.count()

    print(
        f"\nTotal rows: {total_rows:,}"
    )

    if total_rows == 0:

        raise ValueError(
            "Gold V2 dataset is empty."
        )

    print(
        "\nCustomer count:"
    )

    customer_count = (
        df
        .select(
            "customer_unique_id"
        )
        .distinct()
        .count()
    )

    print(
        f"{customer_count:,}"
    )

    print(
        "\nChurn distribution:"
    )

    (
        df
        .groupBy(
            "churn_90d"
        )
        .count()
        .orderBy(
            "churn_90d"
        )
        .show()
    )

    print(
        "\nObservation range:"
    )

    (
        df
        .select(
            min(
                "observation_date"
            ).alias(
                "min_date"
            ),

            max(
                "observation_date"
            ).alias(
                "max_date"
            ),
        )
        .show()
    )

    print(
        "\nCorrected product features:"
    )

    (
        df
        .select(
            "unique_products_180d",
            "unique_categories_180d",
            "unique_sellers_180d",
        )
        .summary()
        .show()
    )

    print(
        "\nFeature correlation sanity:"
    )

    (
        df
        .select(
            "order_count_180d",
            "item_count_180d",
            "unique_products_180d",
            "unique_categories_180d",
            "unique_sellers_180d",
        )
        .summary()
        .show()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    spark = create_spark()

    spark.sparkContext.setLogLevel(
        "WARN"
    )

    print(
        "\n============================================"
    )

    print(
        " OLIST GOLD CUSTOMER CHURN V2"
    )

    print(
        "============================================"
    )

    (
        customers,
        orders,
        order_items,
        payments,
        reviews,
        products,
        category_translation,
    ) = read_silver(
        spark
    )

    print(
        "\nPreparing customers..."
    )

    customers = prepare_customers(
        customers
    )

    print(
        "Preparing completed orders..."
    )

    orders = prepare_orders(
        orders,
        customers,
    )

    print(
        "Preparing item data..."
    )

    items = prepare_items(
        order_items,
        products,
        category_translation,
    )

    print(
        "Preparing order-level item features..."
    )

    item_features = (
        prepare_item_order_features(
            items
        )
    )

    print(
        "Preparing payment features..."
    )

    payment_features = (
        prepare_payment_features(
            payments
        )
    )

    print(
        "Preparing review features..."
    )

    review_features = (
        prepare_review_features(
            reviews
        )
    )

    print(
        "Building order-level table..."
    )

    order_level = (
        build_order_level_table(
            orders,
            item_features,
            payment_features,
            review_features,
        )
    )

    print(
        f"Order-level rows: "
        f"{order_level.count():,}"
    )

    print(
        "Creating customer observations..."
    )

    observations = (
        create_observation_dates(
            order_level
        )
    )

    print(
        f"Observation rows: "
        f"{observations.count():,}"
    )

    print(
        "Building corrected 180-day features..."
    )

    features = (
        build_features(
            order_level,
            observations,
            items,
        )
    )

    print(
        f"Feature rows: "
        f"{features.count():,}"
    )

    print(
        "Creating 90-day churn labels..."
    )

    gold = (
        build_churn_label(
            order_level,
            features,
        )
    )

    gold = (
        gold
        .dropDuplicates(
            [
                "customer_unique_id",
                "observation_date",
            ]
        )
    )

    print(
        "Cleaning Gold dataset..."
    )

    gold = clean_gold_dataset(
        gold
    )

    validate_gold(
        gold
    )

    print(
        "\nWriting Gold V2..."
    )

    (
        gold

        .repartition(
            "observation_year",
            "observation_month",
        )

        .write

        .mode(
            "overwrite"
        )

        .partitionBy(
            "observation_year",
            "observation_month",
        )

        .parquet(
            GOLD_PATH
        )
    )

    print(
        "\n============================================"
    )

    print(
        " GOLD V2 CREATED SUCCESSFULLY"
    )

    print(
        "============================================"
    )

    print(
        f"Location: {GOLD_PATH}"
    )

    spark.stop()


if __name__ == "__main__":

    main()