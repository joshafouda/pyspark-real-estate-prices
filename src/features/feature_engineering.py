from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.sql.functions import (
    col, when, count, lit, expr, log, round as spark_round,
    percent_rank, sum as spark_sum, udf, sqrt, pow, asin, sin, cos, rand
)
from pyspark.sql.types import DoubleType, FloatType
import numpy as np
import sys
from pathlib import Path

# Ajout du chemin src au PYTHONPATH
src_path = str(Path(__file__).resolve().parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from utils.spark_utils import create_spark_session

class FeatureEngineering:
    def __init__(self, spark_session=None):
        """Initialise la classe de feature engineering.
        
        Args:
            spark_session: Session Spark à utiliser. Si None, une nouvelle session sera créée.
        """
        self.spark = spark_session if spark_session else create_spark_session()
        self.fitted_pipeline = None
        self.capping_values = {}
        
        # Coordonnées de la Tour Eiffel
        self.EIFFEL_LAT = 48.858370
        self.EIFFEL_LON = 2.294481
        
        # Seuil pour les catégories rares
        self.RARE_CATEGORY_THRESHOLD = 0.05
        
        self.CATEGORICAL_COLS = []
        self.NUMERIC_COLS = []

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
        """Crée les features dérivées.
        
        Args:
            df: DataFrame Spark avec les données brutes
        
        Returns:
            DataFrame Spark avec les features dérivées ajoutées
        """
        # Calculer la distance à la Tour Eiffel
        df = df.withColumn(
            "distance_to_eiffel",
            self._haversine_distance(
                col("approximate_latitude"), 
                col("approximate_longitude"),
                lit(self.EIFFEL_LAT),
                lit(self.EIFFEL_LON)
            )
        )
        
        # Calculer le nombre de pièces par m²
        df = df.withColumn(
            "rooms_per_m2",
            col("nb_rooms") / col("size")
        )
        
        # Identifier les maisons (vs appartements)
        df = df.withColumn(
            "is_house",
            when(col("property_type").isin(["house", "villa"]), 1).otherwise(0)
        )
        
        # Log transformation pour les variables avec une distribution asymétrique
        if "size" in df.columns:
            df = df.withColumn("log_size", log(col("size")))
        if "land_size" in df.columns:
            df = df.withColumn("log_land_size", log(col("land_size")))
        
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
        """Ajuste le pipeline de transformation et transforme les données.
        
        Args:
            df: DataFrame Spark avec les données brutes
        
        Returns:
            DataFrame Spark avec les données transformées
        """
        # Créer les features dérivées
        df = self._create_derived_features(df)
        
        # Plafonner les valeurs aberrantes pour les colonnes numériques
        # Exclure 'price' car c'est notre target
        numeric_cols_to_cap = [col for col in self.NUMERIC_COLS if col != 'price']
        for column in numeric_cols_to_cap:
            if column in df.columns:  # Vérifier si la colonne existe
                df = self._cap_outliers(df, column)
        
        # Créer le pipeline de transformation
        stages = []
        
        # Indexer et encoder les variables catégorielles
        for column in self.CATEGORICAL_COLS:
            if column in df.columns:  # Vérifier si la colonne existe
                # Indexer
                indexer = StringIndexer(
                    inputCol=column,
                    outputCol=f"{column}_indexed",
                    handleInvalid="keep"
                )
                stages.append(indexer)
                
                # Encoder
                encoder = OneHotEncoder(
                    inputCols=[f"{column}_indexed"],
                    outputCols=[f"{column}_encoded"],
                    dropLast=True
                )
                stages.append(encoder)
        
        # Sélectionner les colonnes pour le vecteur de features
        feature_cols = [
            # Colonnes numériques (sauf price)
            *[col for col in self.NUMERIC_COLS if col != 'price'],
            # Features dérivées
            'distance_to_eiffel', 'rooms_per_m2', 'is_house',
            'log_size', 'log_land_size',
            # Colonnes encodées
            *[f"{col}_encoded" for col in self.CATEGORICAL_COLS]
        ]
        
        # Ajouter l'assembleur de features
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features",
            handleInvalid="skip"
        )
        stages.append(assembler)
        
        # Créer et ajuster le pipeline
        self.fitted_pipeline = Pipeline(stages=stages).fit(df)
        
        # Transformer les données
        return self.fitted_pipeline.transform(df)
    
    def transform(self, df):
        """Transforme de nouvelles données en utilisant le pipeline ajusté.
        
        Args:
            df: DataFrame Spark avec les données brutes
        
        Returns:
            DataFrame Spark avec les données transformées
        """
        if self.fitted_pipeline is None:
            raise ValueError("Le pipeline n'a pas encore été ajusté. Utilisez fit_transform d'abord.")
        
        # Créer les features dérivées
        df = self._create_derived_features(df)
        
        # Plafonner les valeurs aberrantes pour les colonnes numériques
        # Exclure 'price' car c'est notre target et il peut ne pas être présent
        numeric_cols_to_cap = [col for col in self.NUMERIC_COLS if col != 'price']
        for column in numeric_cols_to_cap:
            if column in df.columns:  # Vérifier si la colonne existe
                df = self._cap_outliers(df, column)
        
        return self.fitted_pipeline.transform(df)

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
    parser.add_argument('--input-batch', type=str, default='data/raw/X_test.csv',
                       help='Chemin vers les données pour les prédictions batch (default: data/raw/X_test.csv)')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                       help='Dossier de sortie pour les données transformées (default: data/processed)')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Dossier pour sauvegarder le pipeline (default: models)')
    
    args = parser.parse_args()

    try:
        # Créer une session Spark
        spark = create_spark_session("RealEstate_FeatureEngineering")
        
        # Initialiser le feature engineering
        fe = FeatureEngineering(spark)
        
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

        # Créer les dossiers de sortie
        output_dir = Path(args.output_dir)
        model_dir = Path(args.model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Charger les données d'entraînement
        logger.info("Chargement des données d'entraînement...")
        train_path = Path(args.input_train)
        if not train_path.exists():
            raise FileNotFoundError(f"Le fichier d'entraînement n'existe pas : {train_path}")
            
        df = spark.read.parquet(str(train_path))
        
        # Diviser en train/validation si le fichier validation n'existe pas
        validation_path = Path(args.input_validation)
        if not validation_path.exists():
            logger.info("Création des jeux d'entraînement et de validation (80/20)...")
            train_ratio = 0.8
            seed = 42
            
            # Créer une colonne pour la division aléatoire
            df = df.withColumn("rand", rand(seed=seed))
            
            # Diviser les données
            df_train = df.where(f"rand <= {train_ratio}").drop("rand")
            df_validation = df.where(f"rand > {train_ratio}").drop("rand")
            
            # Sauvegarder le jeu de validation
            logger.info("Sauvegarde du jeu de validation...")
            df_validation.write.mode("overwrite").parquet(str(validation_path))
            
            # Vérifier les tailles
            train_count = df_train.count()
            val_count = df_validation.count()
            total = train_count + val_count
            logger.info(f"Split effectué - Total: {total:,} -> Train: {train_count:,} ({train_count/total:.1%}), "
                       f"Validation: {val_count:,} ({val_count/total:.1%})")
        else:
            logger.info("Les fichiers train et validation existent déjà.")
            logger.info("Pour recréer le split, supprimez le fichier validation.parquet")
            logger.info(f"Chargement du jeu de validation depuis {validation_path}...")
            df_validation = spark.read.parquet(str(validation_path))
            
            # Identifier les IDs dans la validation pour les exclure du train
            validation_ids = set(df_validation.select("id_annonce").toPandas()["id_annonce"])
            df_train = df.where(~col("id_annonce").isin(validation_ids))
        
        train_count = df_train.count()
        val_count = df_validation.count()
        total = train_count + val_count
        logger.info(f"Distribution des données - Total: {total:,} -> Train: {train_count:,} ({train_count/total:.1%}), "
                   f"Validation: {val_count:,} ({val_count/total:.1%})")
        
        logger.info(f"Nombre d'observations - Train: {df_train.count():,}, Validation: {df_validation.count():,}")
        
        # Transformation des données d'entraînement
        logger.info("Transformation des données d'entraînement...")
        df_train_transformed = fe.fit_transform(df_train)
        
        # Sauvegarder les données d'entraînement transformées
        logger.info("Sauvegarde des données d'entraînement transformées...")
        df_train_transformed.write.mode("overwrite").parquet(
            str(output_dir / "train_transformed.parquet")
        )

        # Transformer et sauvegarder les données de validation
        logger.info("Transformation des données de validation...")
        df_validation_transformed = fe.transform(df_validation)
        
        logger.info("Sauvegarde des données de validation transformées...")
        df_validation_transformed.write.mode("overwrite").parquet(
            str(output_dir / "validation_transformed.parquet")
        )

        # Transformer les données pour les prédictions batch si elles existent
        batch_path = Path(args.input_batch)
        if batch_path.exists():
            logger.info("Chargement et transformation des données pour les prédictions batch...")
            df_batch = spark.read.csv(str(batch_path), header=True, inferSchema=True)
            df_batch_transformed = fe.transform(df_batch)
            
            logger.info("Sauvegarde des données batch transformées...")
            df_batch_transformed.write.mode("overwrite").parquet(
                str(output_dir / "batch_transformed.parquet")
            )
            logger.info(f"Nombre d'observations dans le jeu batch : {df_batch.count():,}")
        else:
            logger.warning(f"Le fichier pour les prédictions batch n'existe pas : {batch_path}")

        # Sauvegarder le pipeline et ses paramètres
        logger.info("Sauvegarde du pipeline...")
        pipeline_path = model_dir / "feature_engineering_pipeline"
        if pipeline_path.exists():
            import shutil
            shutil.rmtree(str(pipeline_path))
        fe.fitted_pipeline.save(str(pipeline_path))

        params = {
            'CATEGORICAL_COLS': fe.CATEGORICAL_COLS,
            'NUMERIC_COLS': fe.NUMERIC_COLS,
            'RARE_CATEGORY_THRESHOLD': fe.RARE_CATEGORY_THRESHOLD,
            'EIFFEL_LAT': fe.EIFFEL_LAT,
            'EIFFEL_LON': fe.EIFFEL_LON,
            'capping_values': fe.capping_values
        }

        with open(model_dir / "feature_engineering_params.json", 'w') as f:
            json.dump(params, f, indent=4)

        logger.info("Processus de feature engineering terminé avec succès!")

    except Exception as e:
        logger.error(f"Une erreur s'est produite : {str(e)}")
        raise

    finally:
        # Nettoyage des ressources
        if 'fe' in locals():
            fe.cleanup()
        elif 'spark' in locals():
            spark.stop()