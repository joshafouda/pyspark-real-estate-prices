# %% [markdown]
# # Analyse Exploratoire des Données - Real Estate Price Prediction
# 
# Ce notebook présente l'analyse exploratoire des données d'entraînement pour le projet de prédiction des prix immobiliers.

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
import pyspark.sql.functions as F
from pyspark.sql.functions import col, count, isnan, when, mean, stddev, min, max
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Configuration de matplotlib
#plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (12, 8)

# Création du dossier pour sauvegarder les figures
FIGURES_DIR = project_dir / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def save_plot(name):
    """Sauvegarde le graphique actuel"""
    plt.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches='tight')

# Initialisation de Spark avec la configuration optimisée
spark = create_spark_session("RealEstate_EDA")

# %% [markdown]
# ## 2. Chargement des données

# %%
# Chargement des données d'entraînement
X_train = spark.read.csv("../data/raw/X_train.csv", header=True, inferSchema=True)
y_train = spark.read.csv("../data/raw/y_train.csv", header=True, inferSchema=True)

# Joindre X_train et y_train
df_train = X_train.join(y_train, "id_annonce")

# Afficher les premières lignes
df_train.show(5)

# %% [markdown]
# ## 3. Analyse de la structure des données

# %%
# Afficher le schéma des données
print("Schema des données:")
df_train.printSchema()

# Nombre de lignes et colonnes
print(f"\nNombre de lignes: {df_train.count()}")
print(f"Nombre de colonnes: {len(df_train.columns)}")

# %% [markdown]
# ## 4. Analyse des valeurs manquantes

# %%
# Calculer le nombre de valeurs manquantes par colonne
def missing_values_analysis(df):
    missing_counts = []
    total_rows = df.count()
    
    for col_name in df.columns:
        missing_count = df.filter(col(col_name).isNull()).count()
        missing_percentage = (missing_count / total_rows) * 100
        missing_counts.append({
            'column': col_name,
            'missing_count': missing_count,
            'missing_percentage': missing_percentage
        })
    
    return pd.DataFrame(missing_counts)

missing_df = missing_values_analysis(df_train)
missing_df = missing_df.sort_values('missing_percentage', ascending=False)
print("\nAnalyse des valeurs manquantes:")
print(missing_df)

# Visualisation des valeurs manquantes
plt.figure(figsize=(15, 6))
plt.bar(missing_df['column'], missing_df['missing_percentage'])
plt.xticks(rotation=45, ha='right')
plt.title('Pourcentage de valeurs manquantes par colonne')
plt.ylabel('Pourcentage de valeurs manquantes')
plt.tight_layout()
save_plot("missing_values")
plt.show()

# %% [markdown]
# ## 5. Analyse de la distribution du prix (variable cible)

# %%
# Convertir en pandas pour la visualisation
prices = df_train.select('price').toPandas()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(data=prices, x='price', bins=50)
plt.title('Distribution des prix')
plt.xlabel('Prix (€)')

plt.subplot(1, 2, 2)
sns.histplot(data=prices, x='price', bins=50, log_scale=True)
plt.title('Distribution des prix (échelle log)')
plt.xlabel('Prix (€)')
plt.tight_layout()
save_plot("price_distribution")
plt.show()

# Statistiques descriptives des prix
price_stats = df_train.select(
    mean('price').alias('mean'),
    stddev('price').alias('std'),
    min('price').alias('min'),
    max('price').alias('max')
).toPandas()
print("\nStatistiques descriptives des prix:")
print(price_stats)

# %% [markdown]
# ## 6. Analyse des variables catégorielles

# %%
categorical_cols = ['property_type', 'energy_performance_category', 
                   'ghg_category', 'exposition']

for col_name in categorical_cols:
    # Remplacer les valeurs None par "Unknown"
    df_temp = df_train.withColumn(
        col_name,
        F.coalesce(F.col(col_name), F.lit("Unknown"))
    )
    
    # Distribution des valeurs
    value_counts = df_temp.groupBy(col_name) \
        .count() \
        .orderBy('count', ascending=False) \
        .toPandas()
    
    plt.figure(figsize=(10, 5))
    plt.bar(value_counts[col_name], value_counts['count'])
    plt.title(f'Distribution de {col_name}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_plot(f"distribution_{col_name}")
    plt.show()
    
    # Prix moyen par catégorie
    avg_price = df_temp.groupBy(col_name) \
        .agg(mean('price').alias('avg_price')) \
        .orderBy('avg_price', ascending=False) \
        .toPandas()
    
    plt.figure(figsize=(10, 5))
    plt.bar(avg_price[col_name], avg_price['avg_price'])
    plt.title(f'Prix moyen par {col_name}')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Prix moyen (€)')
    plt.tight_layout()
    save_plot(f"avg_price_by_{col_name}")
    plt.show()

# %% [markdown]
# ## 7. Analyse des variables numériques

# %%
numeric_cols = ['size', 'floor', 'land_size', 'energy_performance_value',
                'ghg_value', 'nb_rooms', 'nb_bedrooms', 'nb_bathrooms',
                'nb_parking_places', 'nb_boxes', 'nb_photos']

# Statistiques descriptives
numeric_stats = df_train.select([
    mean(col).alias(f'mean_{col}') for col in numeric_cols
] + [
    stddev(col).alias(f'std_{col}') for col in numeric_cols
] + [
    min(col).alias(f'min_{col}') for col in numeric_cols
] + [
    max(col).alias(f'max_{col}') for col in numeric_cols
]).toPandas()

print("Statistiques descriptives des variables numériques:")
print(numeric_stats)

# Distribution et relation avec le prix
for col_name in numeric_cols:
    data = df_train.select(col_name, 'price').toPandas()
    
    plt.figure(figsize=(15, 5))
    
    # Distribution
    plt.subplot(1, 2, 1)
    sns.histplot(data=data, x=col_name, bins=50)
    plt.title(f'Distribution de {col_name}')
    
    # Relation avec le prix
    plt.subplot(1, 2, 2)
    plt.scatter(data[col_name], data['price'], alpha=0.5)
    plt.title(f'Relation entre {col_name} et le prix')
    plt.xlabel(col_name)
    plt.ylabel('Prix (€)')
    
    plt.tight_layout()
    save_plot(f"analysis_{col_name}")
    plt.show()

# %% [markdown]
# ## 8. Analyse géographique

# %%
# Visualisation de la distribution géographique des biens
geo_data = df_train.select('approximate_latitude', 'approximate_longitude', 'price').toPandas()

plt.figure(figsize=(12, 8))
scatter = plt.scatter(geo_data['approximate_longitude'], 
                     geo_data['approximate_latitude'],
                     c=geo_data['price'],
                     cmap='viridis',
                     alpha=0.5)
plt.colorbar(scatter, label='Prix (€)')
plt.title('Distribution géographique des biens et leurs prix')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
save_plot("geographic_distribution")
plt.show()

# %% [markdown]
# ## 9. Analyse des corrélations

# %%
# Sélection des variables numériques pour la matrice de corrélation
numeric_features = ['price', 'size', 'floor', 'land_size', 
                   'energy_performance_value', 'ghg_value', 'nb_rooms', 
                   'nb_bedrooms', 'nb_bathrooms', 'nb_parking_places', 
                   'nb_boxes', 'nb_photos']

# Conversion en pandas pour le calcul des corrélations
corr_data = df_train.select(numeric_features).toPandas()
corr_matrix = corr_data.corr()

# Visualisation de la matrice de corrélation
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Matrice de corrélation des variables numériques')
plt.tight_layout()
save_plot("correlation_matrix")
plt.show()

# %% [markdown]
# ## 10. Conclusions et insights
