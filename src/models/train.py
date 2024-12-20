from pyspark.sql import SparkSession
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import joblib
import os

class ModelTrainer:
    def __init__(self, spark_session=None):
        """Initialize Model Trainer"""
        self.spark = spark_session or SparkSession.builder \
            .appName("Real Estate Model Training") \
            .getOrCreate()
        self.model = None
        
    def train(self, train_data, **params):
        """Train the model with given parameters"""
        # Initialize the model with parameters
        self.model = RandomForestRegressor(
            featuresCol="features",
            labelCol="price",
            **params
        )
        
        # Fit the model
        self.fitted_model = self.model.fit(train_data)
        return self.fitted_model
    
    def evaluate(self, test_data):
        """Evaluate the model on test data"""
        if self.fitted_model is None:
            raise ValueError("Model not trained. Call train first.")
        
        # Make predictions
        predictions = self.fitted_model.transform(test_data)
        
        # Evaluate predictions
        evaluator = RegressionEvaluator(
            labelCol="price",
            predictionCol="prediction",
            metricName="rmse"
        )
        
        rmse = evaluator.evaluate(predictions)
        r2 = evaluator.setMetricName("r2").evaluate(predictions)
        mae = evaluator.setMetricName("mae").evaluate(predictions)
        
        return {
            "rmse": rmse,
            "r2": r2,
            "mae": mae
        }
    
    def save_model(self, path):
        """Save the trained model"""
        if self.fitted_model is None:
            raise ValueError("Model not trained. Call train first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        self.fitted_model.save(path)
        
    def load_model(self, path):
        """Load a trained model"""
        self.fitted_model = RandomForestRegressor.load(path)
        return self.fitted_model
