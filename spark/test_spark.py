import os
import sys
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext

os.environ['PYSPARK_PYTHON'] = sys.executable


spark = SparkSession.builder \
    .appName("test_spark") \
    .master("spark://master:7077") \
    .getOrCreate()

data = [("James", "Smith", "USA", "CA"),
        ("Michael", "Rose", "USA", "NY"),
        ("Robert", "Williams", "USA", "CA"),
        ("Maria", "Jones", "USA", "FL")]

columns = ["firstname", "lastname", "country", "state"]
df = spark.createDataFrame(data=data, schema=columns)
df.show()