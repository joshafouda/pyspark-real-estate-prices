# %% [markdown]
# # Test du Pipeline de Feature Engineering
# 
# Ce notebook teste notre pipeline de feature engineering sur les données d'entraînement.

# %% [markdown]
# ## 1. Configuration de l'environnement

# %%
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ajouter le répertoire src au PYTHONPATH
notebook_dir = Path(__file__).parent.absolute()
project_dir = notebook_dir.parent
sys.path.append(str(project_dir))

from src.utils.spark_utils import create_spark_session
from src.features.feature_engineering import FeatureEngineering
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, isnan, when, mean, stddev

# Configuration de matplotlib
plt.rcParams['figure.figsize'] = (12, 8)

# Création du dossier pour sauvegarder les figures
FIGURES_DIR = project_dir / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def save_plot(name):
    """Sauvegarde le graphique actuel"""
    plt.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches='tight')
    plt.close()

# %% [markdown]
# ## 2. Chargement des données et initialisation de Spark

# %%
try:
    # Initialisation de Spark
    spark = create_spark_session("RealEstate_FeatureEngineering_Test")

    # Charger les données d'entraînement
    train_path = project_dir / "data" / "raw" / "train.parquet"
    df_train = spark.read.parquet(str(train_path))

    print("Schéma des données :")
    df_train.printSchema()

    print("\nAperçu des colonnes disponibles :")
    print(df_train.columns)

    print(f"\nNombre d'observations : {df_train.count():,}")
    print(f"Nombre de colonnes : {len(df_train.columns):,}")

    # %% [markdown]
    # ## 3. Test du Pipeline de Feature Engineering

    # %%
    # Instanciation de la classe FeatureEngineering
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

    # Application du pipeline complet
    print("Application du pipeline de feature engineering...")
    df_transformed = fe.fit_transform(df_train)
    print("Transformation terminée !")

    # %% [markdown]
    # ## 4. Analyse des Features Transformées

    # %%
    # Afficher les colonnes disponibles après transformation
    print("Features disponibles après transformation :")
    for column in df_transformed.columns:
        print(f"- {column}")

    # %% [markdown]
    # ### 4.1 Analyse des Features Numériques

    # %%
    # Sélectionner les features numériques importantes
    numeric_features = [col for col in [
        'size', 'price_per_m2', 'rooms_per_m2', 'distance_to_eiffel',
        'log_size', 'log_land_size'
    ] if col in df_transformed.columns]

    # Convertir en Pandas pour la visualisation
    df_numeric = df_transformed.select(numeric_features).toPandas()

    # Créer un histogramme pour chaque feature numérique
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for idx, feature in enumerate(numeric_features):
        if idx < len(axes):
            sns.histplot(data=df_numeric, x=feature, ax=axes[idx])
            axes[idx].set_title(f'Distribution de {feature}')

    plt.tight_layout()
    save_plot("numeric_features_distribution")

    # %% [markdown]
    # ### 4.2 Analyse des Features Catégorielles

    # %%
    # Sélectionner les features catégorielles encodées
    categorical_features = [col for col in fe.CATEGORICAL_COLS if col in df_transformed.columns]

    # Analyser la distribution des catégories
    for feature in categorical_features:
        print(f"\nDistribution de {feature}:")
        df_transformed.groupBy(feature) \
            .agg(count("*").alias("count")) \
            .orderBy("count", ascending=False) \
            .show()

    # %% [markdown]
    # ### 4.3 Analyse des Valeurs Manquantes

    # %%
    def missing_values_analysis(df):
        """Analyse des valeurs manquantes dans le DataFrame"""
        missing_counts = []
        total_rows = df.count()
        
        for column_name in df.columns:
            missing_count = df.filter(col(column_name).isNull()).count()
            missing_percentage = (missing_count / total_rows) * 100
            
            missing_counts.append({
                'column': column_name,
                'missing_count': missing_count,
                'missing_percentage': missing_percentage
            })
        
        return pd.DataFrame(missing_counts)

    # Analyser les valeurs manquantes
    missing_df = missing_values_analysis(df_transformed)
    missing_df = missing_df.sort_values('missing_percentage', ascending=False)

    # Afficher les résultats
    print("Analyse des valeurs manquantes :")
    print(missing_df)

    # Visualiser les valeurs manquantes
    plt.figure(figsize=(12, 6))
    sns.barplot(data=missing_df.head(10), 
                x='column', 
                y='missing_percentage')
    plt.xticks(rotation=45)
    plt.title('Top 10 des colonnes avec le plus de valeurs manquantes')
    plt.tight_layout()
    save_plot("missing_values_analysis")

    # %% [markdown]
    # ## 5. Sauvegarde des Données Transformées et du Pipeline

    # %%
    # Créer le dossier pour les données transformées
    processed_data_dir = project_dir / "data" / "processed"
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    # Sauvegarder les données transformées
    print("Sauvegarde des données transformées...")
    df_transformed.write.mode("overwrite").parquet(str(processed_data_dir / "train_transformed.parquet"))
    print("Données transformées sauvegardées !")

    # Créer le dossier pour les modèles
    models_dir = project_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Sauvegarder le pipeline
    print("\nSauvegarde du pipeline...")
    pipeline_path = models_dir / "feature_engineering_pipeline"
    fe.fitted_pipeline.save(str(pipeline_path))

    # Sauvegarder les paramètres du pipeline
    import json
    params = {
        'CATEGORICAL_COLS': fe.CATEGORICAL_COLS,
        'NUMERIC_COLS': fe.NUMERIC_COLS,
        'RARE_CATEGORY_THRESHOLD': fe.RARE_CATEGORY_THRESHOLD,
        'EIFFEL_LAT': fe.EIFFEL_LAT,
        'EIFFEL_LON': fe.EIFFEL_LON,
        'capping_values': fe.capping_values
    }

    with open(models_dir / "feature_engineering_params.json", 'w') as f:
        json.dump(params, f, indent=4)
    print("Pipeline et paramètres sauvegardés !")

    # %% [markdown]
    # ## 6. Test de Chargement et Validation

    # %%
    # Test de chargement des données transformées
    print("Test de chargement des données transformées...")
    df_loaded = spark.read.parquet(str(processed_data_dir / "train_transformed.parquet"))
    print(f"Nombre de lignes chargées : {df_loaded.count():,}")
    print(f"Nombre de colonnes : {len(df_loaded.columns):,}")

    # Test de chargement du pipeline
    from pyspark.ml import PipelineModel
    print("\nTest de chargement du pipeline...")
    loaded_pipeline = PipelineModel.load(str(pipeline_path))

    # Charger les paramètres
    with open(models_dir / "feature_engineering_params.json", 'r') as f:
        loaded_params = json.load(f)
    print("Pipeline et paramètres chargés avec succès !")

    # Vérifier que tout est bien sauvegardé
    print("\nVérification des paramètres sauvegardés :")
    for key, value in loaded_params.items():
        print(f"{key}: {value}")

finally:
    # Nettoyage des ressources Spark
    if 'fe' in locals():
        fe.cleanup()
    elif 'spark' in locals():
        spark.stop()
