from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql.functions import (
    col, when, count, lit, expr, log, round as spark_round,
    percent_rank, sum as spark_sum, udf, sqrt, pow, asin, sin, cos, rand
)
from pyspark.sql.types import DoubleType, FloatType
import numpy as np
import sys
from pathlib import Path
import json

# Ajout du chemin src au PYTHONPATH
src_path = str(Path(__file__).resolve().parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from utils.spark_utils import create_spark_session

class FeatureEngineering:
    def __init__(self, spark_session=None, strict_mode=True):
        """Initialise la classe de feature engineering.
        
        Args:
            spark_session: Session Spark à utiliser. Si None, une nouvelle session sera créée.
            strict_mode: Si True, élimine les lignes avec des valeurs invalides (pour l'entraînement).
                        Si False, remplace les valeurs invalides par des valeurs par défaut (pour l'inférence).
        """
        if spark_session is None:
            self.spark = create_spark_session()
        else:
            self.spark = spark_session
        self.fitted_pipeline = None
        self.capping_values = {}
        self.strict_mode = strict_mode
        
        # Coordonnées de la Tour Eiffel
        self.EIFFEL_LAT = 48.858370
        self.EIFFEL_LON = 2.294481
        
        # Seuil pour les catégories rares
        self.RARE_CATEGORY_THRESHOLD = 0.05
        
        self.CATEGORICAL_COLS = []
        self.NUMERIC_COLS = []
        
        # Valeurs par défaut pour le mode permissif
        self.default_values = {
            'approximate_latitude': self.EIFFEL_LAT,
            'approximate_longitude': self.EIFFEL_LON,
            'nb_rooms': 0,
            'size': 0,
            'land_size': 0,
            'property_type': 'unknown',
            'energy_performance_category': 'unknown',
            'ghg_category': 'unknown',
            'exposition': 'unknown',
            'floor': 0,
            'energy_performance_value': 0,
            'ghg_value': 0,
            'nb_bedrooms': 0,
            'nb_bathrooms': 0,
            'nb_parking_places': 0,
            'nb_boxes': 0,
            'nb_photos': 0
        }
        
    def cleanup(self):
        """Nettoie les ressources Spark."""
        if self.spark:
            try:
                self.spark.stop()
            except Exception as e:
                print(f"Erreur lors de l'arrêt de la session Spark : {e}")
            finally:
                self.spark = None

    def _create_derived_features(self, df):
        """Crée les features dérivées."""
        # En mode permissif, remplacer les valeurs nulles
        if not self.strict_mode:
            df = df.fillna(self.default_values)
        else:
            # En mode strict, remplacer les valeurs nulles par 0 pour les colonnes numériques
            numeric_cols_to_fill = [col for col in self.NUMERIC_COLS if col != 'price']
            df = df.na.fill({col: 0 for col in numeric_cols_to_fill})
            
            # Pour les colonnes catégorielles, remplacer les valeurs nulles par 'unknown'
            categorical_cols_to_fill = [col for col in self.CATEGORICAL_COLS]
            df = df.na.fill({col: 'unknown' for col in categorical_cols_to_fill})
        
        # Calculer la distance à la Tour Eiffel
        df = df.withColumn(
            "distance_to_eiffel",
            when(
                col("approximate_latitude").isNull() | col("approximate_longitude").isNull(),
                lit(0)
            ).otherwise(
                self._haversine_distance(
                    col("approximate_latitude"), 
                    col("approximate_longitude"),
                    lit(self.EIFFEL_LAT),
                    lit(self.EIFFEL_LON)
                )
            )
        )
        
        # Calculer le nombre de pièces par m²
        df = df.withColumn(
            "rooms_per_m2",
            when(
                col("size").isNull() | (col("size") <= 0),
                lit(0)
            ).otherwise(
                when(col("nb_rooms").isNull(), lit(0)).otherwise(col("nb_rooms") / col("size"))
            )
        )
        
        # Identifier les maisons (vs appartements)
        df = df.withColumn(
            "is_house",
            when(col("property_type").isNull(), lit(0))
            .when(col("property_type").isin(["house", "villa"]), 1)
            .otherwise(0)
        )
        
        # Log transformation pour les variables avec une distribution asymétrique
        if "size" in df.columns:
            df = df.withColumn(
                "log_size",
                when(
                    col("size").isNull() | (col("size") <= 0),
                    lit(0)
                ).otherwise(
                    log(col("size"))
                )
            )
        if "land_size" in df.columns:
            df = df.withColumn(
                "log_land_size",
                when(
                    col("land_size").isNull() | (col("land_size") <= 0),
                    lit(0)
                ).otherwise(
                    log(col("land_size"))
                )
            )
        
        return df

    def _cap_outliers(self, df, column):
        """Plafonne les valeurs aberrantes d'une colonne en utilisant les quantiles.
        
        Args:
            df: DataFrame Spark
            column: Nom de la colonne à plafonner
        
        Returns:
            DataFrame avec les valeurs plafonnées
        """
        # Calculer les quantiles si ce n'est pas déjà fait
        if column not in self.capping_values:
            quantiles = df.approxQuantile(column, [0.01, 0.99], 0.01)
            self.capping_values[column] = {
                'lower': quantiles[0],
                'upper': quantiles[1]
            }
        
        # Plafonner les valeurs
        return df.withColumn(
            column,
            when(
                col(column) > self.capping_values[column]['upper'],
                self.capping_values[column]['upper']
            ).when(
                col(column) < self.capping_values[column]['lower'],
                self.capping_values[column]['lower']
            ).otherwise(col(column))
        )

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calcule la distance en kilomètres entre deux points en utilisant la formule de Haversine.
        
        Args:
            lat1, lon1: Latitude et longitude du premier point
            lat2, lon2: Latitude et longitude du deuxième point
            
        Returns:
            Distance en kilomètres
        """
        # Rayon de la Terre en kilomètres
        R = lit(6371.0)
        
        # Conversion en radians
        lat1_rad = lat1 * lit(np.pi/180)
        lon1_rad = lon1 * lit(np.pi/180)
        lat2_rad = lat2 * lit(np.pi/180)
        lon2_rad = lon2 * lit(np.pi/180)
        
        # Différences de latitude et longitude
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        # Formule de Haversine
        a = pow(sin(dlat/2), 2) + cos(lat1_rad) * cos(lat2_rad) * pow(sin(dlon/2), 2)
        c = lit(2) * asin(sqrt(a))
        
        return R * c

    def fit_transform(self, df):
        """Ajuste le pipeline de transformation et transforme les données."""
        # Créer les features dérivées
        df = self._create_derived_features(df)
        
        # Plafonner les valeurs aberrantes pour les colonnes numériques
        numeric_cols_to_cap = [col for col in self.NUMERIC_COLS if col != 'price']
        for column in numeric_cols_to_cap:
            if column in df.columns:
                df = self._cap_outliers(df, column)
        
        # Créer le pipeline de transformation
        stages = []
        
        # Indexer et encoder les variables catégorielles
        for column in self.CATEGORICAL_COLS:
            if column in df.columns:
                # Indexer
                indexer = StringIndexer(
                    inputCol=column,
                    outputCol=f"{column}_indexed",
                    handleInvalid="skip" if self.strict_mode else "keep"
                )
                stages.append(indexer)
                
                # Encoder
                encoder = OneHotEncoder(
                    inputCols=[f"{column}_indexed"],
                    outputCols=[f"{column}_encoded"],
                    dropLast=True,
                    handleInvalid="error" if self.strict_mode else "keep"
                )
                stages.append(encoder)
        
        # Sélectionner les colonnes pour le vecteur de features
        numeric_features = [col for col in self.NUMERIC_COLS if col != 'price' and col in df.columns]
        derived_features = ['distance_to_eiffel', 'rooms_per_m2', 'is_house', 'log_size', 'log_land_size']
        categorical_features = [f"{col}_encoded" for col in self.CATEGORICAL_COLS if col in df.columns]
        
        # Vérifier que toutes les colonnes existent
        feature_cols = []
        for col in numeric_features + derived_features + categorical_features:
            if col in df.columns or any(stage.getOutputCol() == col for stage in stages):
                feature_cols.append(col)
        
        # Ajouter l'assembleur de features
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features",
            handleInvalid="keep"  # "keep" pour garder les valeurs nulles
        )
        stages.append(assembler)
        
        # Créer et ajuster le pipeline
        self.fitted_pipeline = Pipeline(stages=stages).fit(df)
        
        # Transformer les données
        return self.fitted_pipeline.transform(df)
    
    def transform(self, df):
        """Transforme de nouvelles données en utilisant le pipeline ajusté."""
        if self.fitted_pipeline is None:
            raise ValueError("Le pipeline n'a pas encore été ajusté. Utilisez fit_transform d'abord.")
        
        # Créer les features dérivées
        df = self._create_derived_features(df)
        
        # Plafonner les valeurs aberrantes pour les colonnes numériques
        numeric_cols_to_cap = [col for col in self.NUMERIC_COLS if col != 'price']
        for column in numeric_cols_to_cap:
            if column in df.columns:
                df = self._cap_outliers(df, column)
        
        return self.fitted_pipeline.transform(df)

    def save(self, path: str):
        """Sauvegarde le pipeline et les paramètres.
        
        Args:
            path: Chemin du dossier où sauvegarder le pipeline et les paramètres
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder le pipeline
        if self.fitted_pipeline:
            self.fitted_pipeline.write().overwrite().save(str(path / "pipeline"))
        else:
            raise ValueError("Le pipeline n'a pas encore été ajusté")
        
        # Sauvegarder les paramètres
        params = {
            "capping_values": self.capping_values,
            "CATEGORICAL_COLS": self.CATEGORICAL_COLS,
            "NUMERIC_COLS": self.NUMERIC_COLS,
            "EIFFEL_LAT": self.EIFFEL_LAT,
            "EIFFEL_LON": self.EIFFEL_LON,
            "RARE_CATEGORY_THRESHOLD": self.RARE_CATEGORY_THRESHOLD
        }
        
        with open(path / "params.json", "w") as f:
            json.dump(params, f, indent=4)
    
    def load(self, path: str):
        """Charge le pipeline et les paramètres.
        
        Args:
            path: Chemin du dossier contenant le pipeline et les paramètres
        """
        path = Path(path)
        
        # Charger le pipeline
        if (path / "pipeline").exists():
            self.fitted_pipeline = PipelineModel.load(str(path / "pipeline"))
        else:
            raise ValueError(f"Pipeline non trouvé dans {path}")
        
        # Charger les paramètres
        if (path / "params.json").exists():
            with open(path / "params.json", "r") as f:
                params = json.load(f)
            
            self.capping_values = params.get("capping_values", {})
            self.CATEGORICAL_COLS = params.get("CATEGORICAL_COLS", [])
            self.NUMERIC_COLS = params.get("NUMERIC_COLS", [])
            self.EIFFEL_LAT = params.get("EIFFEL_LAT", 48.858370)
            self.EIFFEL_LON = params.get("EIFFEL_LON", 2.294481)
            self.RARE_CATEGORY_THRESHOLD = params.get("RARE_CATEGORY_THRESHOLD", 0.05)
        else:
            raise ValueError(f"Paramètres non trouvés dans {path}")

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import json
    import logging

    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Parser les arguments
    parser = argparse.ArgumentParser(description='Feature Engineering pour les données immobilières')
    parser.add_argument('--input-train', type=str, default='data/raw/train.parquet',
                       help='Chemin vers les données d\'entraînement (default: data/raw/train.parquet)')
    parser.add_argument('--input-validation', type=str, default='data/raw/validation.parquet',
                       help='Chemin vers les données de validation (default: data/raw/validation.parquet)')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                       help='Dossier de sortie pour les données transformées (default: data/processed)')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Dossier pour sauvegarder le pipeline (default: models)')
    
    args = parser.parse_args()

    try:
        # Créer une session Spark
        spark = create_spark_session("RealEstate_FeatureEngineering")
        
        # Initialiser le feature engineering
        logger.info("Initialisation du feature engineering...")
        fe = FeatureEngineering(spark_session=spark, strict_mode=True)
        
        # Définir les colonnes catégorielles et numériques
        fe.CATEGORICAL_COLS = [
            'property_type', 
            'energy_performance_category',
            'ghg_category', 
            'exposition'
        ]
        
        fe.NUMERIC_COLS = [
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

        # Charger et transformer les données
        logger.info("Chargement des données d'entraînement...")
        train_data = spark.read.parquet(args.input_train)
        
        logger.info("Chargement des données de validation...")
        validation_data = spark.read.parquet(args.input_validation)

        # Transformation des données
        logger.info("Application du feature engineering sur les données d'entraînement...")
        train_transformed = fe.fit_transform(train_data)
        
        logger.info("Application du feature engineering sur les données de validation...")
        validation_transformed = fe.transform(validation_data)

        # Créer les dossiers de sortie
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_dir = Path(args.model_dir) / "feature_engineering"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Sauvegarder les données transformées
        logger.info("Sauvegarde des données transformées...")
        train_transformed.write.mode("overwrite").parquet(str(output_dir / "train_transformed.parquet"))
        validation_transformed.write.mode("overwrite").parquet(str(output_dir / "validation_transformed.parquet"))

        # Sauvegarder le pipeline
        logger.info("Sauvegarde du pipeline de feature engineering...")
        fe.save(str(model_dir))

        logger.info("Feature engineering terminé avec succès!")

    except Exception as e:
        logger.error(f"Une erreur s'est produite : {str(e)}")
        raise

    finally:
        if 'fe' in locals():
            fe.cleanup()
        if 'spark' in locals():
            spark.stop()