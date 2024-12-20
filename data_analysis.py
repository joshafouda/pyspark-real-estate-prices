from pyspark.sql import SparkSession
from pyspark.sql.types import *

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Real Estate Data Analysis") \
    .getOrCreate()

# Path to data files
data_path = "data/raw/"
files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_random.csv"]

# Analyze each dataset
for file in files:
    print(f"\n{'='*50}")
    print(f"Analyzing {file}")
    print('='*50)
    
    # Read CSV file
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(data_path + file)
    
    # Display schema
    print("\nSchema:")
    df.printSchema()
    
    # Display basic statistics
    print("\nBasic Statistics:")
    print(f"Number of rows: {df.count()}")
    print(f"Number of columns: {len(df.columns)}")
    
    # Sample few rows
    print("\nSample of data (5 rows):")
    df.show(5, truncate=False)

# Stop Spark session
spark.stop()
