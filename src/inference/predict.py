from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pandas as pd

class PricePredictor:
    def __init__(self, feature_engineering, model, spark_session=None):
        """Initialize Price Predictor"""
        self.spark = spark_session or SparkSession.builder \
            .appName("Real Estate Price Prediction") \
            .getOrCreate()
        self.feature_engineering = feature_engineering
        self.model = model
        
    def predict_batch(self, input_df):
        """Predict prices for a batch of properties"""
        # Transform features
        features_df = self.feature_engineering.transform(input_df)
        
        # Make predictions
        predictions = self.model.transform(features_df)
        
        # Select only necessary columns
        result_df = predictions.select("id_annonce", "prediction") \
                             .withColumnRenamed("prediction", "price")
        
        return result_df
    
    def predict_single(self, property_dict):
        """Predict price for a single property"""
        # Create schema for single property
        schema = StructType([
            StructField("id_annonce", IntegerType(), True),
            StructField("property_type", StringType(), True),
            StructField("approximate_latitude", DoubleType(), True),
            StructField("approximate_longitude", DoubleType(), True),
            StructField("city", StringType(), True),
            StructField("postal_code", IntegerType(), True),
            StructField("size", DoubleType(), True),
            StructField("floor", DoubleType(), True),
            StructField("land_size", DoubleType(), True),
            StructField("energy_performance_value", DoubleType(), True),
            StructField("energy_performance_category", StringType(), True),
            StructField("ghg_value", DoubleType(), True),
            StructField("ghg_category", StringType(), True),
            StructField("exposition", StringType(), True),
            StructField("nb_rooms", DoubleType(), True),
            StructField("nb_bedrooms", DoubleType(), True),
            StructField("nb_bathrooms", DoubleType(), True),
            StructField("nb_parking_places", DoubleType(), True),
            StructField("nb_boxes", DoubleType(), True),
            StructField("nb_photos", DoubleType(), True),
            StructField("has_a_balcony", DoubleType(), True),
            StructField("nb_terraces", DoubleType(), True),
            StructField("has_a_cellar", DoubleType(), True),
            StructField("has_a_garage", DoubleType(), True),
            StructField("has_air_conditioning", DoubleType(), True),
            StructField("last_floor", DoubleType(), True),
            StructField("upper_floors", DoubleType(), True)
        ])
        
        # Convert single property to DataFrame
        property_df = self.spark.createDataFrame([property_dict], schema)
        
        # Get prediction
        prediction = self.predict_batch(property_df)
        
        # Convert to pandas and get the price
        price = prediction.toPandas()["price"].iloc[0]
        
        return price
