from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    lit,
    max as spark_max,
    min as spark_min,
    when,
)

GOLD_PATH = "hdfs:///data/gold/olist/customer_churn_features"

ML_PATH = "hdfs:///data/gold/olist/ml_dataset"


def create_spark():

    return (
        SparkSession.builder
        .appName("OlistPrepareMLDataset")
        .config("spark.sql.ansi.enabled", "true")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def main():

    spark = create_spark()

    spark.sparkContext.setLogLevel("WARN")

    print("\n============================================")
    print(" PREPARING ML DATASET")
    print("============================================")

    df = spark.read.parquet(GOLD_PATH)

    print(f"\nOriginal rows: {df.count():,}")

    # -------------------------------------------------
    # 1. Determine the latest observation date
    # -------------------------------------------------

    date_range = (
        df.select(
            spark_min("observation_date").alias("min_date"),
            spark_max("observation_date").alias("max_date"),
        )
        .collect()[0]
    )

    min_date = date_range["min_date"]
    max_date = date_range["max_date"]

    print(f"Original minimum date: {min_date}")
    print(f"Original maximum date: {max_date}")

    # -------------------------------------------------
    # 2. Require a complete 90-day future window
    #
    # We use the maximum observation date as the
    # boundary and remove the final 90 days.
    # -------------------------------------------------

    cutoff_date = spark.sql(
        f"""
        SELECT date_sub(
            DATE('{max_date}'),
            90
        ) AS cutoff
        """
    ).collect()[0]["cutoff"]

    print(f"Maximum valid observation date: {cutoff_date}")

    df = df.filter(
        col("observation_date") <= lit(cutoff_date)
    )

    print(
        f"Rows after 90-day boundary filter: "
        f"{df.count():,}"
    )

    # -------------------------------------------------
    # 3. Remove raw timestamp columns
    #
    # These are useful for diagnostics but aren't
    # required as model predictors.
    # -------------------------------------------------

    timestamp_columns = [
        "last_purchase_timestamp",
        "first_purchase_timestamp_180d",
    ]

    for c in timestamp_columns:

        if c in df.columns:
            df = df.drop(c)

    # -------------------------------------------------
    # 4. Remove partition helper columns
    #
    # They represent the observation date and should
    # not become predictive features.
    # -------------------------------------------------

    partition_columns = [
        "observation_year",
        "observation_month",
    ]

    for c in partition_columns:

        if c in df.columns:
            df = df.drop(c)

    # -------------------------------------------------
    # 5. Remove rows without a valid target
    # -------------------------------------------------

    df = df.filter(
        col("churn_90d").isNotNull()
    )

    # -------------------------------------------------
    # 6. Cast target explicitly
    # -------------------------------------------------

    df = df.withColumn(
        "churn_90d",
        col("churn_90d").cast("integer"),
    )

    # -------------------------------------------------
    # 7. Add temporal split indicator
    #
    # This is intentionally chronological.
    #
    # 70% earliest observations → train
    # 15% next observations     → validation
    # 15% latest observations   → test
    # -------------------------------------------------

    dates = (
        df.select("observation_date")
        .distinct()
        .orderBy("observation_date")
    )

    date_list = [
        row["observation_date"]
        for row in dates.collect()
    ]

    n_dates = len(date_list)

    train_end_index = int(n_dates * 0.70)
    validation_end_index = int(n_dates * 0.85)

    train_end_date = date_list[
        train_end_index - 1
    ]

    validation_end_date = date_list[
        validation_end_index - 1
    ]

    print("\nChronological split:")
    print(
        f"Train end:       {train_end_date}"
    )
    print(
        f"Validation end:  {validation_end_date}"
    )

    df = (
        df
        .withColumn(
            "dataset_split",
            when(
                col("observation_date")
                <= lit(train_end_date),
                lit("train"),
            )
            .when(
                col("observation_date")
                <= lit(validation_end_date),
                lit("validation"),
            )
            .otherwise(
                lit("test")
            ),
        )
    )

    # -------------------------------------------------
    # 8. Validate split
    # -------------------------------------------------

    print("\nSplit distribution:")

    (
        df.groupBy("dataset_split")
        .count()
        .orderBy("dataset_split")
        .show()
    )

    print("\nTarget distribution by split:")

    (
        df.groupBy(
            "dataset_split",
            "churn_90d",
        )
        .count()
        .orderBy(
            "dataset_split",
            "churn_90d",
        )
        .show()
    )

    print("\nObservation ranges by split:")

    (
        df.groupBy("dataset_split")
        .agg(
            spark_min("observation_date").alias(
                "min_date"
            ),
            spark_max("observation_date").alias(
                "max_date"
            ),
        )
        .orderBy("dataset_split")
        .show()
    )

    # -------------------------------------------------
    # 9. Save
    # -------------------------------------------------

    (
        df
        .repartition(
            "dataset_split"
        )
        .write
        .mode("overwrite")
        .partitionBy(
            "dataset_split"
        )
        .parquet(ML_PATH)
    )

    print("\n============================================")
    print(" ML DATASET CREATED")
    print("============================================")
    print(f"Location: {ML_PATH}")

    spark.stop()


if __name__ == "__main__":
    main()