from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("EnterpriseMLPlatformTest")
    .master("local[*]")
    .getOrCreate()
)

data = [
    (1, "Alice", 100),
    (2, "Bob", 200),
    (3, "Charlie", 300),
]

df = spark.createDataFrame(
    data,
    ["id", "customer", "revenue"]
)

df.show()

print("Row count:", df.count())

spark.stop()
