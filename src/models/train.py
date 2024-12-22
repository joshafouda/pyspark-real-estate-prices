"""
Module pour l'entraînement des modèles de prédiction de prix immobiliers.
"""
from pathlib import Path
import json
from typing import Dict, Any, Optional, Union
from datetime import datetime

from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.regression import RandomForestRegressor, LinearRegression, GBTRegressor, RandomForestRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from src.features.target_transformer import TargetTransformer
from src.features.feature_engineering import FeatureEngineering

class ModelTrainer:
    """Classe pour l'entraînement et l'évaluation des modèles."""
    
    AVAILABLE_MODELS = {
        "rf": RandomForestRegressor,
        "lr": LinearRegression,
        "gbt": GBTRegressor
    }
    
    def __init__(self, spark_session: Optional[SparkSession] = None):
        """Initialise le Model Trainer.
        
        Args:
            spark_session: Session Spark existante ou None pour en créer une nouvelle
        """
        self.spark = spark_session or SparkSession.builder \
            .appName("Real Estate Model Training") \
            .getOrCreate()
        
        self.model = None
        self.fitted_model = None
        self.feature_engineering = None
        self.target_transformer = None
        self.feature_cols = None
        self.best_params = None
        
    def prepare_data(self, 
                    train_data: DataFrame,
                    validation_data: Optional[DataFrame] = None,
                    target_col: str = "price") -> tuple:
        """Prépare les données pour l'entraînement.
        
        Args:
            train_data: DataFrame d'entraînement
            validation_data: DataFrame de validation (optionnel)
            target_col: Nom de la colonne cible
            
        Returns:
            tuple: (train_prepared, validation_prepared)
        """
        # Initialiser et appliquer la transformation sur la variable cible
        self.target_transformer = TargetTransformer(target_col=target_col)
        train_data = self.target_transformer.fit_transform(train_data)
        
        if validation_data is not None:
            validation_data = self.target_transformer.transform(validation_data)
        
        return train_data, validation_data
        
    def train(self, 
              train_data: DataFrame,
              model_type: str = "rf",
              params: Optional[Dict[str, Any]] = None,
              use_cv: bool = True,
              cv_folds: int = 5) -> Any:
        """Entraîne le modèle avec les paramètres donnés.
        
        Args:
            train_data: DataFrame d'entraînement
            model_type: Type de modèle ('rf', 'lr', ou 'gbt')
            params: Paramètres du modèle
            use_cv: Utiliser la validation croisée
            cv_folds: Nombre de folds pour la CV
            
        Returns:
            Modèle entraîné
        """
        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(f"Type de modèle inconnu: {model_type}")
        
        # Initialiser le modèle
        base_params = {
            "featuresCol": "features",
            "labelCol": "price",  # Utilisation directe de la colonne price
            **(params or {})
        }
        
        self.model = self.AVAILABLE_MODELS[model_type](**base_params)
        
        if use_cv:
            # Définir la grille de paramètres pour la CV
            param_grid = self._get_param_grid(model_type)
            
            # Configurer la validation croisée
            evaluator = RegressionEvaluator(
                labelCol="price",  # Utilisation directe de la colonne price
                predictionCol="prediction",
                metricName="rmse"
            )
            
            cv = CrossValidator(
                estimator=self.model,
                estimatorParamMaps=param_grid,
                evaluator=evaluator,
                numFolds=cv_folds
            )
            
            # Entraîner avec CV
            cv_model = cv.fit(train_data)
            self.fitted_model = cv_model.bestModel
            # Extraction des meilleurs paramètres de façon sécurisée
            param_map = cv_model.bestModel.extractParamMap()
            self.best_params = {str(param.name): value for param, value in param_map.items()}
        else:
            # Entraînement simple
            self.fitted_model = self.model.fit(train_data)
            
        return self.fitted_model
    
    def evaluate(self, test_data: DataFrame) -> Dict[str, float]:
        """Évalue le modèle sur les données de test.
        
        Args:
            test_data: DataFrame de test
            
        Returns:
            Dict avec les métriques d'évaluation
        """
        if not self.fitted_model:
            raise ValueError("Le modèle n'a pas encore été entraîné")
        
        # Faire les prédictions
        predictions = self.fitted_model.transform(test_data)
        
        # Calculer les métriques
        evaluator = RegressionEvaluator(labelCol="price", 
                                      predictionCol="prediction")
        
        metrics = {}
        for metric in ["rmse", "r2", "mae"]:
            evaluator.setMetricName(metric)
            metrics[metric] = evaluator.evaluate(predictions)
            
        return metrics
    
    def save_model(self, model_dir: Union[str, Path], overwrite: bool = False):
        """Sauvegarde le modèle entraîné et ses métadonnées.
        
        Args:
            model_dir: Chemin du dossier où sauvegarder le modèle
            overwrite: Option pour écraser le modèle existant
        """
        if not self.fitted_model:
            raise ValueError("Le modèle n'a pas encore été entraîné")
        
        # Créer le dossier si nécessaire
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder le modèle
        if overwrite:
            self.fitted_model.write().overwrite().save(str(model_dir / "model"))
        else:
            self.fitted_model.save(str(model_dir / "model"))
        
        # Sauvegarder le pipeline de feature engineering
        fe_pipeline_dir = model_dir / "feature_engineering"
        if fe_pipeline_dir.exists() and overwrite:
            import shutil
            shutil.rmtree(str(fe_pipeline_dir))
        self.feature_engineering.save(str(fe_pipeline_dir))
        
        # Sauvegarder le transformateur de target
        self.target_transformer.save(str(model_dir / "target_transformer.json"))
        
        # Sauvegarder les métadonnées
        metadata = {
            "model_type": self.fitted_model.__class__.__name__,
            "training_date": datetime.now().isoformat(),
            "feature_columns": self.feature_engineering.NUMERIC_COLS + self.feature_engineering.CATEGORICAL_COLS,
            "target_column": "price",
            "best_params": self.best_params if self.best_params else {},
            "CATEGORICAL_COLS": self.feature_engineering.CATEGORICAL_COLS,
            "NUMERIC_COLS": self.feature_engineering.NUMERIC_COLS
        }
        
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
    
    def load_model(self, model_dir: Union[str, Path]) -> None:
        """Charge un modèle entraîné et ses métadonnées.
        
        Args:
            model_dir: Chemin du dossier contenant le modèle
        """
        from pyspark.ml.regression import RandomForestRegressionModel
        
        model_dir = Path(model_dir)
        
        # Charger le modèle
        self.fitted_model = RandomForestRegressionModel.load(str(model_dir / "model"))
        
        # Charger le pipeline de feature engineering
        self.feature_engineering = FeatureEngineering(strict_mode=False)  # Mode permissif pour l'inférence
        self.feature_engineering.load(str(model_dir / "feature_engineering"))
        
        # Charger les métadonnées
        with open(model_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
            self.best_params = metadata.get("best_params", {})
            
            # Charger les colonnes depuis les métadonnées
            self.feature_engineering.CATEGORICAL_COLS = metadata.get("CATEGORICAL_COLS", [])
            self.feature_engineering.NUMERIC_COLS = metadata.get("NUMERIC_COLS", [])
    
    def _get_param_grid(self, model_type: str) -> list:
        """Retourne la grille de paramètres pour la validation croisée.
        
        Args:
            model_type: Type de modèle
            
        Returns:
            Liste de paramètres pour ParamGridBuilder
        """
        if model_type == "rf":
            return ParamGridBuilder() \
                .addGrid(self.model.numTrees, [10, 50]) \
                .addGrid(self.model.maxDepth, [5, 10]) \
                .addGrid(self.model.minInstancesPerNode, [2, 4]) \
                .build()
        elif model_type == "gbt":
            return ParamGridBuilder() \
                .addGrid(self.model.maxDepth, [5, 10]) \
                .addGrid(self.model.maxIter, [10, 50]) \
                .build()
        else:  # linear regression
            return ParamGridBuilder() \
                .addGrid(self.model.regParam, [0.1, 1.0]) \
                .addGrid(self.model.elasticNetParam, [0.0, 1.0]) \
                .build()

    def cleanup(self):
        """Nettoie les ressources Spark."""
        if self.spark:
            try:
                self.spark.stop()
            except Exception:
                pass  # Ignorer les erreurs de fermeture
            finally:
                self.spark = None

    def __del__(self):
        """Destructeur de la classe."""
        self.cleanup()

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import logging
    from src.utils.spark_utils import create_spark_session

    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Parser les arguments
    parser = argparse.ArgumentParser(description='Entraînement des modèles')
    parser.add_argument('--input-train', type=str, default='data/raw/train.parquet',
                       help='Chemin vers les données d\'entraînement')
    parser.add_argument('--input-validation', type=str, default='data/raw/validation.parquet',
                       help='Chemin vers les données de validation')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Dossier de sortie pour les modèles')
    parser.add_argument('--model-type', type=str, default='rf',
                       choices=['rf', 'lr', 'gbt'],
                       help='Type de modèle à entraîner')
    parser.add_argument('--overwrite', action='store_true',
                       help='Écraser le modèle existant')
    
    args = parser.parse_args()

    try:
        # Créer une session Spark
        spark = create_spark_session("RealEstate_ModelTraining")
        
        # Initialiser le trainer
        trainer = ModelTrainer(spark_session=spark)
        
        # Charger les données
        logger.info("Chargement des données...")
        train_data = spark.read.parquet(args.input_train)
        validation_data = spark.read.parquet(args.input_validation)
        
        # Initialiser le feature engineering
        logger.info("Initialisation du feature engineering...")
        trainer.feature_engineering = FeatureEngineering(spark_session=spark, strict_mode=True)
        trainer.feature_engineering.CATEGORICAL_COLS = [
            'property_type', 
            'energy_performance_category',
            'ghg_category', 
            'exposition'
        ]
        
        trainer.feature_engineering.NUMERIC_COLS = [
            'size',
            'price',
            'land_size',
            'nb_rooms',
            'floor', 
            'energy_performance_value', 
            'ghg_value',
            'nb_bedrooms', 
            'nb_bathrooms', 
            'nb_parking_places',
            'nb_boxes', 
            'nb_photos'
        ]
        
        # Transformation des données
        logger.info("Application du feature engineering...")
        train_transformed = trainer.feature_engineering.fit_transform(train_data)
        validation_transformed = trainer.feature_engineering.transform(validation_data)
        
        # Préparation des données (transformation de la cible)
        logger.info("Préparation des données...")
        train_prepared, val_prepared = trainer.prepare_data(
            train_data=train_transformed,
            validation_data=validation_transformed
        )
        
        # Entraînement du modèle
        logger.info(f"Entraînement du modèle {args.model_type}...")
        model = trainer.train(
            train_data=train_prepared,
            model_type=args.model_type,
            use_cv=True,
            cv_folds=3  # Réduire le nombre de folds pour la validation croisée
        )
        
        # Évaluation
        logger.info("Évaluation du modèle...")
        metrics = trainer.evaluate(val_prepared)
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Sauvegarde du modèle
        output_dir = Path(args.output_dir)
        model_path = output_dir / args.model_type
        trainer.save_model(model_path, overwrite=args.overwrite)
        logger.info(f"Modèle sauvegardé dans : {model_path}")

    except Exception as e:
        logger.error(f"Une erreur s'est produite : {str(e)}")
        raise

    finally:
        # Nettoyage
        if 'trainer' in locals():
            trainer.cleanup()
        elif 'spark' in locals():
            spark.stop()
