from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when


HDFS_BASE = "hdfs://localhost:9000/data/raw/olist"


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("EnterpriseML-ValidateRaw")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def validate_dataset(spark: SparkSession, filename: str) -> None:
    path = f"{HDFS_BASE}/{filename}"

    print("\n" + "=" * 80)
    print(f"VALIDATING: {filename}")
    print("=" * 80)

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(path)
    )

    print(f"Rows: {df.count()}")
    print(f"Columns: {len(df.columns)}")

    print("\nSchema:")
    df.printSchema()

    print("\nColumns:")
    print(df.columns)

    print("\nNull counts:")
    null_counts = df.select(
        [
            count(
                when(col(c).isNull(), c)
            ).alias(c)
            for c in df.columns
        ]
    )

    null_counts.show(truncate=False)

    print("\nSample records:")
    df.show(5, truncate=False)


def main() -> None:
    spark = create_spark_session()

    datasets = [
        "olist_customers_dataset.csv",
        "olist_geolocation_dataset.csv",
        "olist_order_items_dataset.csv",
        "olist_order_payments_dataset.csv",
        "olist_order_reviews_dataset.csv",
        "olist_orders_dataset.csv",
        "olist_products_dataset.csv",
        "olist_sellers_dataset.csv",
        "product_category_name_translation.csv",
    ]

    try:
        for dataset in datasets:
            validate_dataset(spark, dataset)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()