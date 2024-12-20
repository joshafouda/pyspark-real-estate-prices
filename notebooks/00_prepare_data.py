# %% [markdown]
# # Préparation des données
# 
# Ce notebook fusionne les données X_train.csv et y_train.csv et les convertit en format Parquet.

# %% [markdown]
# ## 1. Configuration de l'environnement

# %%
import sys
import os
from pathlib import Path

# Ajouter le répertoire src au PYTHONPATH
notebook_dir = Path(__file__).parent.absolute()
project_dir = notebook_dir.parent
sys.path.append(str(project_dir))

from src.utils.spark_utils import create_spark_session
from pyspark.sql.functions import col, when

try:
    # Initialisation de Spark
    spark = create_spark_session("RealEstate_DataPrep")

    # %% [markdown]
    # ## 2. Chargement et fusion des données

    # %%
    # Chemins des fichiers
    x_train_path = project_dir / "data" / "raw" / "X_train.csv"
    y_train_path = project_dir / "data" / "raw" / "y_train.csv"
    parquet_path = project_dir / "data" / "raw" / "train.parquet"

    # Vérifier si les fichiers existent
    if not x_train_path.exists():
        raise FileNotFoundError(f"Le fichier {x_train_path} n'existe pas.")
    if not y_train_path.exists():
        raise FileNotFoundError(f"Le fichier {y_train_path} n'existe pas.")

    # Lecture des CSV
    print("Lecture des fichiers CSV...")
    df_x = spark.read.csv(
        str(x_train_path),
        header=True,
        inferSchema=True
    )

    df_y = spark.read.csv(
        str(y_train_path),
        header=True,
        inferSchema=True
    )

    # Fusion des DataFrames
    print("\nFusion des données...")
    df = df_x.join(df_y, "id_annonce")

    # Afficher le schéma inféré
    print("\nSchéma des données :")
    df.printSchema()

    # Conversion des types si nécessaire
    print("\nConversion des types de données...")
    df = df.select([
        col("id_annonce").cast("integer"),
        col("property_type").cast("string"),
        col("approximate_latitude").cast("double"),
        col("approximate_longitude").cast("double"),
        col("city").cast("string"),
        col("postal_code").cast("integer"),
        col("size").cast("double"),
        col("floor").cast("integer"),
        col("land_size").cast("double"),
        col("energy_performance_value").cast("double"),
        col("energy_performance_category").cast("string"),
        col("ghg_value").cast("double"),
        col("ghg_category").cast("string"),
        col("exposition").cast("string"),
        col("nb_rooms").cast("integer"),
        col("nb_bedrooms").cast("integer"),
        col("nb_bathrooms").cast("integer"),
        col("nb_parking_places").cast("integer"),
        col("nb_boxes").cast("integer"),
        col("nb_photos").cast("integer"),
        col("price").cast("double")
    ])

    # Sauvegarde en format Parquet
    print("\nSauvegarde en format Parquet...")
    df.write.mode("overwrite").parquet(str(parquet_path))

    print(f"\nDonnées sauvegardées dans : {parquet_path}")
    print(f"Nombre d'observations : {df.count():,}")
    print(f"Nombre de colonnes : {len(df.columns):,}")

    # %% [markdown]
    # ## 3. Vérification des données

    # %%
    # Lecture du fichier Parquet pour vérification
    df_verify = spark.read.parquet(str(parquet_path))

    # Afficher quelques statistiques
    print("Aperçu des données :")
    df_verify.select([
        col("price"),
        col("size"),
        col("nb_rooms"),
        col("property_type")
    ]).summary().show()

    print("\nDistribution des types de biens :")
    df_verify.groupBy("property_type").count().orderBy("count", ascending=False).show()

    # Vérifier que la jointure n'a pas perdu de données
    print("\nVérification de l'intégrité des données :")
    print(f"Nombre de lignes dans X_train : {df_x.count():,}")
    print(f"Nombre de lignes dans y_train : {df_y.count():,}")
    print(f"Nombre de lignes après fusion : {df_verify.count():,}")

finally:
    # Nettoyage des ressources Spark
    if 'spark' in locals():
        spark.stop()
