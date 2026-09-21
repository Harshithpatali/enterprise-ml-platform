from pyspark.sql import SparkSession


RAW_BASE = "hdfs://localhost:9000/data/raw/olist"
BRONZE_BASE = "hdfs://localhost:9000/data/bronze/olist"


DATASETS = [
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


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("EnterpriseML-BronzeIngestion")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def ingest_dataset(spark: SparkSession, filename: str) -> None:
    source_path = f"{RAW_BASE}/{filename}"

    # Remove .csv extension for the Bronze directory name
    dataset_name = filename.replace(".csv", "")

    destination_path = f"{BRONZE_BASE}/{dataset_name}"

    print("\n" + "=" * 80)
    print(f"BRONZE INGESTION: {filename}")
    print("=" * 80)

    # Read raw CSV
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("mode", "PERMISSIVE")
        .csv(source_path)
    )

    print(f"Source: {source_path}")
    print(f"Rows: {df.count()}")
    print(f"Columns: {len(df.columns)}")

    print("\nSchema:")
    df.printSchema()

    # Write to Bronze as Parquet
    (
        df.write
        .mode("overwrite")
        .parquet(destination_path)
    )

    print(f"\nWritten to: {destination_path}")

    # Verify the written data
    bronze_df = spark.read.parquet(destination_path)

    print(f"Verified Bronze rows: {bronze_df.count()}")
    print(f"Verified Bronze columns: {len(bronze_df.columns)}")


def main() -> None:
    spark = create_spark_session()

    try:
        print("=" * 80)
        print("ENTERPRISE ML PLATFORM")
        print("BRONZE INGESTION PIPELINE")
        print("=" * 80)

        for dataset in DATASETS:
            ingest_dataset(spark, dataset)

        print("\n" + "=" * 80)
        print("BRONZE INGESTION COMPLETED SUCCESSFULLY")
        print("=" * 80)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()