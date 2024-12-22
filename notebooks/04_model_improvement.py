"""
Script pour l'amélioration du modèle de prédiction des prix immobiliers.
"""
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from src.models.train import ModelTrainer
from pyspark.sql import SparkSession
from pyspark.ml.feature import Bucketizer

# Configuration du logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_feature_importance(trainer, feature_names):
    """Analyse et affiche l'importance des features."""
    if not hasattr(trainer.fitted_model, 'featureImportances'):
        logger.info("Ce modèle ne supporte pas l'importance des features")
        return
    
    importances = trainer.fitted_model.featureImportances.toArray()
    # Extraire les noms des features du vecteur de features
    feature_imp = pd.DataFrame({
        'feature': [f"feature_{i}" for i in range(len(importances))],
        'importance': importances
    })
    feature_imp = feature_imp.sort_values('importance', ascending=False)
    
    logger.info("\nTop 10 features les plus importantes:")
    logger.info(feature_imp.head(10))
    
    return feature_imp

def tune_hyperparameters(trainer, train_data):
    """Ajuste les hyperparamètres du Random Forest."""
    from pyspark.ml.regression import RandomForestRegressor
    
    # Création d'un nouveau modèle RF pour l'optimisation
    rf = RandomForestRegressor(labelCol="price", featuresCol="features")
    
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [50, 100, 200]) \
        .addGrid(rf.maxDepth, [10, 15, 20]) \
        .addGrid(rf.minInstancesPerNode, [1, 2, 4]) \
        .build()
    
    evaluator = RegressionEvaluator(
        labelCol="price",
        predictionCol="prediction",
        metricName="rmse")
    
    cv = CrossValidator(estimator=rf,
                       estimatorParamMaps=paramGrid,
                       evaluator=evaluator,
                       numFolds=5)
    
    logger.info("Début de l'optimisation des hyperparamètres...")
    cv_model = cv.fit(train_data)
    
    best_model = cv_model.bestModel
    logger.info("\nMeilleurs paramètres trouvés:")
    logger.info(f"Nombre d'arbres: {best_model._java_obj.getNumTrees()}")
    logger.info(f"Profondeur maximale: {best_model._java_obj.getMaxDepth()}")
    logger.info(f"Instances minimales par noeud: {best_model._java_obj.getMinInstancesPerNode()}")
    
    return best_model

def handle_outliers(spark, data):
    """Traite les cas extrêmes dans les données."""
    # Calcul des quantiles pour le prix
    quantiles = data.select("price").approxQuantile("price", [0.01, 0.99], 0.01)
    lower_bound, upper_bound = quantiles[0], quantiles[1]
    
    # Création de buckets pour les prix extrêmes
    splits = [float('-inf'), lower_bound, upper_bound, float('inf')]
    bucketizer = Bucketizer(splits=splits,
                           inputCol="price",
                           outputCol="price_bucket")
    
    # Application du bucketizer
    bucketed_data = bucketizer.transform(data)
    
    # Filtrage des données
    filtered_data = bucketed_data.filter(
        (bucketed_data.price_bucket == 1.0)  # Garder seulement les prix dans les limites
    ).drop("price_bucket")
    
    initial_count = data.count()
    filtered_count = filtered_data.count()
    removed = initial_count - filtered_count
    
    logger.info(f"\nDonnées filtrées:")
    logger.info(f"Nombre initial d'observations: {initial_count}")
    logger.info(f"Observations supprimées: {removed}")
    logger.info(f"Observations restantes: {filtered_count}")
    
    return filtered_data

if __name__ == "__main__":
    try:
        # Initialisation
        root_dir = Path(__file__).parent.parent
        data_dir = root_dir / "data/processed"
        model_dir = root_dir / "models"
        
        # Création de la session Spark
        spark = SparkSession.builder \
            .appName("Real Estate Model Improvement") \
            .getOrCreate()
        
        # Chargement des données transformées
        trainer = ModelTrainer(spark)
        train_data = spark.read.parquet(str(data_dir / "train_transformed.parquet"))
        
        # Vérification des colonnes
        logger.info("\nColonnes disponibles:")
        logger.info(train_data.columns)
        logger.info(f"Nombre d'observations: {train_data.count()}")
        
        # Initialisation du modèle Random Forest
        trainer.train(train_data, model_type="rf", use_cv=False)
        
        # 1. Traitement des cas extrêmes
        logger.info("\n=== Traitement des cas extrêmes ===")
        cleaned_data = handle_outliers(spark, train_data)
        
        # 2. Analyse de l'importance des features
        logger.info("\n=== Analyse de l'importance des features ===")
        feature_names = [col for col in train_data.columns if col != "price"]
        feature_importance = analyze_feature_importance(trainer, feature_names)
        
        # 3. Optimisation des hyperparamètres
        logger.info("\n=== Optimisation des hyperparamètres ===")
        best_model = tune_hyperparameters(trainer, cleaned_data)
        
        # Sauvegarde du modèle optimisé
        trainer.fitted_model = best_model
        trainer.save_model(model_dir / "rf_optimized")
        
        # Évaluation finale
        metrics = trainer.evaluate(cleaned_data)
        logger.info("\n=== Métriques finales ===")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value}")
            
    except Exception as e:
        logger.error(f"Une erreur s'est produite : {str(e)}")
        raise
        
    finally:
        if 'spark' in locals():
            spark.stop()
