from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("OlistHDFSTest")
    .master("local[*]")
    .getOrCreate()
)

path = "hdfs://localhost:9000/data/raw/olist/olist_orders_dataset.csv"

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(path)
)

print("Schema:")
df.printSchema()

print("Number of rows:", df.count())

print("Sample records:")
df.show(10, truncate=False)

spark.stop()
