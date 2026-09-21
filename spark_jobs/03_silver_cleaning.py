from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    lower,
    upper,
    trim,
    to_timestamp,
    expr,
)


# ============================================================
# HDFS LOCATIONS
# ============================================================

BRONZE_BASE = "hdfs://localhost:9000/data/bronze/olist"
SILVER_BASE = "hdfs://localhost:9000/data/silver/olist"


# ============================================================
# SPARK SESSION
# ============================================================

def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("EnterpriseML-SilverCleaning")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.ansi.enabled", "true")
        .getOrCreate()
    )


# ============================================================
# GENERAL HELPERS
# ============================================================

def standardize_columns(df):
    """
    Standardize column names to lowercase snake_case.
    """

    for old_name in df.columns:

        new_name = (
            old_name
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        if old_name != new_name:
            df = df.withColumnRenamed(
                old_name,
                new_name
            )

    return df


def safe_timestamp(column_name):
    """
    Safely parse timestamps.

    Invalid timestamp values become NULL rather than
    crashing the Spark job.
    """

    return expr(
        f"try_cast(`{column_name}` AS timestamp)"
    )


def safe_integer(column_name):
    """
    Safely convert a column to integer.

    Invalid values become NULL.
    """

    return expr(
        f"try_cast(`{column_name}` AS int)"
    )


def safe_double(column_name):
    """
    Safely convert a column to double.

    Invalid values become NULL.
    """

    return expr(
        f"try_cast(`{column_name}` AS double)"
    )


def clean_string(column_name):
    """
    Trim whitespace and convert the column to string.
    """

    return trim(
        col(column_name).cast("string")
    )


def write_silver(df, dataset_name):
    """
    Write a cleaned dataframe to the Silver layer.
    """

    output = f"{SILVER_BASE}/{dataset_name}"

    (
        df.write
        .mode("overwrite")
        .parquet(output)
    )

    print(f"\nSilver dataset: {dataset_name}")
    print(f"Output: {output}")
    print(f"Rows: {df.count()}")
    print(f"Columns: {len(df.columns)}")

    return output


# ============================================================
# CUSTOMERS
# ============================================================

def clean_customers(spark):

    path = f"{BRONZE_BASE}/olist_customers_dataset"

    df = spark.read.parquet(path)

    df = standardize_columns(df)

    df = (
        df
        .withColumn(
            "customer_id",
            clean_string("customer_id")
        )
        .withColumn(
            "customer_unique_id",
            clean_string("customer_unique_id")
        )
        .withColumn(
            "customer_zip_code_prefix",
            safe_integer("customer_zip_code_prefix")
        )
        .withColumn(
            "customer_city",
            lower(
                clean_string("customer_city")
            )
        )
        .withColumn(
            "customer_state",
            upper(
                clean_string("customer_state")
            )
        )
        .filter(
            col("customer_id").isNotNull()
        )
        .dropDuplicates(
            ["customer_id"]
        )
    )

    return write_silver(
        df,
        "customers"
    )


# ============================================================
# ORDERS
# ============================================================

def clean_orders(spark):

    path = f"{BRONZE_BASE}/olist_orders_dataset"

    df = spark.read.parquet(path)

    df = standardize_columns(df)

    timestamp_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for c in timestamp_columns:

        df = df.withColumn(
            c,
            safe_timestamp(c)
        )

    df = (
        df
        .withColumn(
            "order_id",
            clean_string("order_id")
        )
        .withColumn(
            "customer_id",
            clean_string("customer_id")
        )
        .withColumn(
            "order_status",
            lower(
                clean_string("order_status")
            )
        )
        .filter(
            col("order_id").isNotNull()
        )
        .dropDuplicates(
            ["order_id"]
        )
    )

    return write_silver(
        df,
        "orders"
    )


# ============================================================
# ORDER ITEMS
# ============================================================

def clean_order_items(spark):

    path = f"{BRONZE_BASE}/olist_order_items_dataset"

    df = spark.read.parquet(path)

    df = standardize_columns(df)

    df = (
        df
        .withColumn(
            "order_id",
            clean_string("order_id")
        )
        .withColumn(
            "order_item_id",
            safe_integer("order_item_id")
        )
        .withColumn(
            "product_id",
            clean_string("product_id")
        )
        .withColumn(
            "seller_id",
            clean_string("seller_id")
        )
        .withColumn(
            "shipping_limit_date",
            safe_timestamp("shipping_limit_date")
        )
        .withColumn(
            "price",
            safe_double("price")
        )
        .withColumn(
            "freight_value",
            safe_double("freight_value")
        )
        .filter(
            col("order_id").isNotNull()
        )
        .dropDuplicates(
            [
                "order_id",
                "order_item_id"
            ]
        )
    )

    return write_silver(
        df,
        "order_items"
    )


# ============================================================
# PAYMENTS
# ============================================================

def clean_payments(spark):

    path = f"{BRONZE_BASE}/olist_order_payments_dataset"

    df = spark.read.parquet(path)

    df = standardize_columns(df)

    df = (
        df
        .withColumn(
            "order_id",
            clean_string("order_id")
        )
        .withColumn(
            "payment_sequential",
            safe_integer("payment_sequential")
        )
        .withColumn(
            "payment_type",
            lower(
                clean_string("payment_type")
            )
        )
        .withColumn(
            "payment_installments",
            safe_integer("payment_installments")
        )
        .withColumn(
            "payment_value",
            safe_double("payment_value")
        )
        .filter(
            col("order_id").isNotNull()
        )
        .dropDuplicates(
            [
                "order_id",
                "payment_sequential"
            ]
        )
    )

    return write_silver(
        df,
        "payments"
    )


# ============================================================
# REVIEWS
# ============================================================

def clean_reviews(spark):

    path = f"{BRONZE_BASE}/olist_order_reviews_dataset"

    df = spark.read.parquet(path)

    df = standardize_columns(df)

    # IMPORTANT:
    # Bronze review_score is STRING.
    # We safely convert it to integer.
    #
    # Bronze timestamp fields are also STRING.
    # We safely convert them to TIMESTAMP.

    df = (
        df
        .withColumn(
            "review_id",
            clean_string("review_id")
        )
        .withColumn(
            "order_id",
            clean_string("order_id")
        )
        .withColumn(
            "review_score",
            safe_integer("review_score")
        )
        .withColumn(
            "review_creation_date",
            safe_timestamp("review_creation_date")
        )
        .withColumn(
            "review_answer_timestamp",
            safe_timestamp("review_answer_timestamp")
        )
        .withColumn(
            "review_comment_title",
            clean_string("review_comment_title")
        )
        .withColumn(
            "review_comment_message",
            clean_string("review_comment_message")
        )
        .filter(
            col("review_id").isNotNull()
        )
        .dropDuplicates(
            ["review_id"]
        )
    )

    return write_silver(
        df,
        "reviews"
    )


# ============================================================
# PRODUCTS
# ============================================================

def clean_products(spark):

    path = f"{BRONZE_BASE}/olist_products_dataset"

    df = spark.read.parquet(path)

    df = standardize_columns(df)

    numeric_columns = [
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]

    for c in numeric_columns:

        df = df.withColumn(
            c,
            safe_double(c)
        )

    df = (
        df
        .withColumn(
            "product_id",
            clean_string("product_id")
        )
        .withColumn(
            "product_category_name",
            lower(
                clean_string(
                    "product_category_name"
                )
            )
        )
        .filter(
            col("product_id").isNotNull()
        )
        .dropDuplicates(
            ["product_id"]
        )
    )

    return write_silver(
        df,
        "products"
    )


# ============================================================
# SELLERS
# ============================================================

def clean_sellers(spark):

    path = f"{BRONZE_BASE}/olist_sellers_dataset"

    df = spark.read.parquet(path)

    df = standardize_columns(df)

    df = (
        df
        .withColumn(
            "seller_id",
            clean_string("seller_id")
        )
        .withColumn(
            "seller_zip_code_prefix",
            safe_integer(
                "seller_zip_code_prefix"
            )
        )
        .withColumn(
            "seller_city",
            lower(
                clean_string("seller_city")
            )
        )
        .withColumn(
            "seller_state",
            upper(
                clean_string("seller_state")
            )
        )
        .filter(
            col("seller_id").isNotNull()
        )
        .dropDuplicates(
            ["seller_id"]
        )
    )

    return write_silver(
        df,
        "sellers"
    )


# ============================================================
# GEOLOCATION
# ============================================================

def clean_geolocation(spark):

    path = f"{BRONZE_BASE}/olist_geolocation_dataset"

    df = spark.read.parquet(path)

    df = standardize_columns(df)

    df = (
        df
        .withColumn(
            "geolocation_zip_code_prefix",
            safe_integer(
                "geolocation_zip_code_prefix"
            )
        )
        .withColumn(
            "geolocation_lat",
            safe_double(
                "geolocation_lat"
            )
        )
        .withColumn(
            "geolocation_lng",
            safe_double(
                "geolocation_lng"
            )
        )
        .withColumn(
            "geolocation_city",
            lower(
                clean_string(
                    "geolocation_city"
                )
            )
        )
        .withColumn(
            "geolocation_state",
            upper(
                clean_string(
                    "geolocation_state"
                )
            )
        )
    )

    return write_silver(
        df,
        "geolocation"
    )


# ============================================================
# CATEGORY TRANSLATION
# ============================================================

def clean_category_translation(spark):

    path = (
        f"{BRONZE_BASE}/"
        "product_category_name_translation"
    )

    df = spark.read.parquet(path)

    df = standardize_columns(df)

    df = (
        df
        .withColumn(
            "product_category_name",
            lower(
                clean_string(
                    "product_category_name"
                )
            )
        )
        .withColumn(
            "product_category_name_english",
            lower(
                clean_string(
                    "product_category_name_english"
                )
            )
        )
        .filter(
            col("product_category_name").isNotNull()
        )
        .dropDuplicates(
            ["product_category_name"]
        )
    )

    return write_silver(
        df,
        "category_translation"
    )


# ============================================================
# SILVER VALIDATION
# ============================================================

def validate_silver_dataset(
    spark,
    dataset_name
):

    path = f"{SILVER_BASE}/{dataset_name}"

    df = spark.read.parquet(path)

    print("\n" + "-" * 80)
    print(f"SILVER VALIDATION: {dataset_name}")
    print("-" * 80)

    print(f"Rows: {df.count()}")
    print(f"Columns: {len(df.columns)}")

    df.printSchema()


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    spark = create_spark_session()

    try:

        print("=" * 80)
        print("ENTERPRISE ML PLATFORM")
        print("SILVER CLEANING PIPELINE")
        print("=" * 80)

        # --------------------------------------------------------
        # Clean datasets
        # --------------------------------------------------------

        clean_customers(spark)

        clean_orders(spark)

        clean_order_items(spark)

        clean_payments(spark)

        clean_reviews(spark)

        clean_products(spark)

        clean_sellers(spark)

        clean_geolocation(spark)

        clean_category_translation(spark)

        # --------------------------------------------------------
        # Validate outputs
        # --------------------------------------------------------

        datasets = [
            "customers",
            "orders",
            "order_items",
            "payments",
            "reviews",
            "products",
            "sellers",
            "geolocation",
            "category_translation",
        ]

        print("\n")
        print("=" * 80)
        print("VALIDATING SILVER DATASETS")
        print("=" * 80)

        for dataset in datasets:

            validate_silver_dataset(
                spark,
                dataset
            )

        print("\n")
        print("=" * 80)
        print("SILVER CLEANING COMPLETED SUCCESSFULLY")
        print("=" * 80)

    finally:

        spark.stop()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()