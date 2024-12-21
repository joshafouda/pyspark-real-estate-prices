"""
Module pour l'entraînement des modèles de prédiction de prix immobiliers.
"""
from pathlib import Path
import json
from typing import Dict, Any, Optional, Union

from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.regression import RandomForestRegressor, LinearRegression, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from src.features.target_transformer import TargetTransformer

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
            "labelCol": self.target_transformer.transformed_col,
            **(params or {})
        }
        
        self.model = self.AVAILABLE_MODELS[model_type](**base_params)
        
        if use_cv:
            # Définir la grille de paramètres pour la CV
            param_grid = self._get_param_grid(model_type)
            
            # Configurer la validation croisée
            evaluator = RegressionEvaluator(
                labelCol=self.target_transformer.transformed_col,
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
        if self.fitted_model is None:
            raise ValueError("Modèle non entraîné. Appelez train() d'abord.")
        
        # Faire les prédictions
        predictions = self.fitted_model.transform(test_data)
        
        # Transformer les prédictions en prix réels
        predictions = self.target_transformer.inverse_transform(
            predictions,
            prediction_col="prediction"
        )
        
        # Évaluer les prédictions
        metrics = {}
        evaluator = RegressionEvaluator(
            labelCol=self.target_transformer.target_col,
            predictionCol="prediction_price"
        )
        
        for metric in ["rmse", "r2", "mae"]:
            evaluator.setMetricName(metric)
            metrics[metric] = evaluator.evaluate(predictions)
        
        return metrics
    
    def save_model(self, model_dir: Union[str, Path]) -> None:
        """Sauvegarde le modèle entraîné et ses métadonnées.
        
        Args:
            model_dir: Chemin du dossier où sauvegarder le modèle
        """
        if self.fitted_model is None:
            raise ValueError("Modèle non entraîné. Appelez train() d'abord.")
        
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder le modèle
        self.fitted_model.save(str(model_dir / "model"))
        
        # Sauvegarder le transformateur de variable cible
        self.target_transformer.save(str(model_dir / "target_transformer.json"))
        
        # Sauvegarder les métadonnées
        metadata = {
            "best_params": self.best_params
        }
        
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
    
    def load_model(self, model_dir: Union[str, Path]) -> None:
        """Charge un modèle entraîné et ses métadonnées.
        
        Args:
            model_dir: Chemin du dossier contenant le modèle
        """
        model_dir = Path(model_dir)
        
        # Charger le modèle
        self.fitted_model = RandomForestRegressor.load(str(model_dir / "model"))
        
        # Charger le transformateur de variable cible
        self.target_transformer = TargetTransformer.load(
            str(model_dir / "target_transformer.json")
        )
        
        # Charger les métadonnées
        with open(model_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
            self.best_params = metadata["best_params"]
    
    def _get_param_grid(self, model_type: str) -> list:
        """Retourne la grille de paramètres pour la validation croisée.
        
        Args:
            model_type: Type de modèle
            
        Returns:
            Liste de paramètres pour ParamGridBuilder
        """
        if model_type == "rf":
            return ParamGridBuilder() \
                .addGrid(self.model.numTrees, [10, 50, 100]) \
                .addGrid(self.model.maxDepth, [5, 10, 15]) \
                .addGrid(self.model.minInstancesPerNode, [1, 2, 4]) \
                .build()
        elif model_type == "gbt":
            return ParamGridBuilder() \
                .addGrid(self.model.maxDepth, [5, 10, 15]) \
                .addGrid(self.model.maxIter, [10, 50, 100]) \
                .build()
        else:  # linear regression
            return ParamGridBuilder() \
                .addGrid(self.model.regParam, [0.01, 0.1, 1.0]) \
                .addGrid(self.model.elasticNetParam, [0.0, 0.5, 1.0]) \
                .build()

    def cleanup(self):
        """Nettoie les ressources Spark."""
        if self.spark:
            self.spark.stop()
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
    parser.add_argument('--input-train', type=str, default='data/processed/train_transformed.parquet',
                       help='Chemin vers les données d\'entraînement transformées')
    parser.add_argument('--input-validation', type=str, default='data/processed/validation_transformed.parquet',
                       help='Chemin vers les données de validation transformées')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Dossier de sortie pour les modèles')
    parser.add_argument('--model-type', type=str, default='rf',
                       choices=['rf', 'lr', 'gbt'],
                       help='Type de modèle à entraîner')
    
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
        
        # Préparation des données
        logger.info("Préparation des données...")
        train_prepared, val_prepared = trainer.prepare_data(
            train_data=train_data,
            validation_data=validation_data
        )
        
        # Entraînement du modèle
        logger.info(f"Entraînement du modèle {args.model_type}...")
        model = trainer.train(
            train_data=train_prepared,
            model_type=args.model_type,
            use_cv=True
        )
        
        # Évaluation
        logger.info("Évaluation du modèle...")
        metrics = trainer.evaluate(val_prepared)
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Sauvegarde du modèle
        output_dir = Path(args.output_dir)
        model_path = output_dir / args.model_type
        trainer.save_model(model_path)
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
