# %% [markdown]
# # Entraînement des Modèles de Prédiction
# 
# Ce notebook permet d'expérimenter avec différents modèles de prédiction des prix immobiliers.
# 
# ## Plan
# 1. Chargement des données
# 2. Préparation pour l'entraînement
# 3. Entraînement de différents modèles
# 4. Analyse des performances
# 5. Sélection et sauvegarde du meilleur modèle

# %%
import sys
from pathlib import Path

# Ajouter le répertoire racine au PYTHONPATH
root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.spark_utils import create_spark_session
from src.models.train import ModelTrainer

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Création de la session Spark
spark = create_spark_session()

# %% [markdown]
# ## 1. Chargement des Données
# 
# Nous chargeons les données transformées par notre pipeline de feature engineering.

# %%
try:
    # Chemins des données
    data_dir = root_dir / "data/processed"
    model_dir = root_dir / "models"

    # Chargement des données
    df_train = spark.read.parquet(str(data_dir / "train_transformed.parquet"))
    df_val = spark.read.parquet(str(data_dir / "validation_transformed.parquet"))

    logger.info(f"Nombre d'observations - Train: {df_train.count():,}, Validation: {df_val.count():,}")

    # %% [markdown]
    # ## 2. Préparation pour l'Entraînement
    # 
    # Nous préparons les données pour l'entraînement en transformant la variable cible (log)

    # %%
    # Initialisation du ModelTrainer
    trainer = ModelTrainer(spark_session=spark)

    # Préparation des données
    df_train_prepared, df_val_prepared = trainer.prepare_data(
        train_data=df_train,
        validation_data=df_val
    )

    # Vérifier que les colonnes nécessaires sont présentes
    required_cols = ["features", "price", trainer.target_transformer.transformed_col]
    missing_cols = [col for col in required_cols if col not in df_train_prepared.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes : {missing_cols}")

    logger.info("Features et variable cible prêtes pour l'entraînement")

    # %% [markdown]
    # ## 3. Entraînement des Modèles
    # 
    # Nous allons entraîner plusieurs modèles :
    # - Random Forest (rf)
    # - Linear Regression (lr)
    # - Gradient Boosting Trees (gbt)

    # %%
    # Dictionnaire pour stocker les résultats
    results = {}

    # Entraînement des différents modèles
    for model_type in ["rf", "lr", "gbt"]:
        logger.info(f"\nEntraînement du modèle {model_type}...")
        
        # Entraînement avec validation croisée
        model = trainer.train(
            train_data=df_train_prepared,
            model_type=model_type,
            use_cv=True,
            cv_folds=5
        )
        
        # Évaluation sur l'ensemble de validation
        metrics = trainer.evaluate(df_val_prepared)
        results[model_type] = metrics
        
        logger.info(f"Métriques de {model_type}:")
        for metric, value in metrics.items():
            logger.info(f"- {metric}: {value:.4f}")

    # %% [markdown]
    # ## 4. Analyse des Performances

    # %%
    # Création d'un DataFrame avec les résultats
    results_df = pd.DataFrame(results).round(4)
    logger.info("\nComparaison des modèles:")
    print(results_df)

    # Visualisation des métriques
    plt.figure(figsize=(12, 6))
    results_df.plot(kind='bar')
    plt.title('Comparaison des Modèles')
    plt.xlabel('Métrique')
    plt.ylabel('Valeur')
    plt.xticks(rotation=45)
    plt.legend(title='Modèle')
    plt.tight_layout()
    plt.show()

    # %% [markdown]
    # ### Analyse des Erreurs
    # 
    # Regardons la distribution des erreurs pour le meilleur modèle.

    # %%
    # Sélection du meilleur modèle (basé sur RMSE)
    best_model_type = min(results.items(), key=lambda x: x[1]['rmse'])[0]
    logger.info(f"\nMeilleur modèle : {best_model_type}")

    # Entraînement du meilleur modèle
    best_model = trainer.train(
        train_data=df_train_prepared,
        model_type=best_model_type,
        use_cv=True
    )

    # Prédictions sur la validation
    predictions = best_model.transform(df_val_prepared)
    predictions = trainer.target_transformer.inverse_transform(
        predictions,
        prediction_col="prediction"
    )

    # Conversion en Pandas pour la visualisation
    pred_df = predictions.select(
        "price",
        "prediction_price"
    ).toPandas()

    # Calcul des erreurs relatives
    pred_df['error_percent'] = (
        (pred_df['prediction_price'] - pred_df['price']) / pred_df['price'] * 100
    )

    # Distribution des erreurs
    plt.figure(figsize=(10, 6))
    sns.histplot(data=pred_df, x='error_percent', bins=50)
    plt.title('Distribution des Erreurs de Prédiction (%)')
    plt.xlabel('Erreur (%)')
    plt.ylabel('Nombre de Prédictions')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.show()

    # Statistiques des erreurs
    logger.info("\nStatistiques des erreurs (%):")
    error_stats = pred_df['error_percent'].describe()
    print(error_stats)

    # %% [markdown]
    # ### Importance des Features
    # 
    # Pour Random Forest et GBT, nous pouvons voir l'importance des features.

    # %%
    if best_model_type in ['rf', 'gbt']:
        # Récupération de l'importance des features
        importances = best_model.featureImportances.toArray()
        logger.info(f"\nNombre de features : {len(importances)}")
        
        # Création d'un DataFrame pour l'importance
        feature_importance = pd.DataFrame({
            'feature_index': range(len(importances)),
            'importance': importances
        })
        
        # Tri et affichage
        feature_importance = feature_importance.sort_values(
            'importance', 
            ascending=False
        ).reset_index(drop=True)
        
        # Top 15 features
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=feature_importance.head(15),
            x='importance',
            y='feature_index'
        )
        plt.title('Top 15 Features les Plus Importantes (par index)')
        plt.xlabel('Importance')
        plt.ylabel('Index de la Feature')
        plt.tight_layout()
        plt.show()

    # %% [markdown]
    # ## 5. Sauvegarde du Meilleur Modèle

    # %%
    # Sauvegarde du modèle et de ses métadonnées
    model_path = model_dir / best_model_type
    trainer.save_model(model_path)
    logger.info(f"\nMeilleur modèle sauvegardé dans : {model_path}")

    # Sauvegarde des résultats de comparaison
    results_df.to_csv(model_dir / "model_comparison.csv")
    logger.info("Comparaison des modèles sauvegardée dans : model_comparison.csv")

    # %% [markdown]
    # ## Conclusion
    # 
    # Nous avons :
    # 1. Entraîné et comparé plusieurs modèles
    # 2. Analysé leurs performances
    # 3. Sauvegardé le meilleur modèle
    # 
    # Le modèle {best_model_type} a obtenu les meilleures performances avec :
    # - RMSE : {results[best_model_type]['rmse']:.4f}
    # - R² : {results[best_model_type]['r2']:.4f}
    # - MAE : {results[best_model_type]['mae']:.4f}

except Exception as e:
    logger.error(f"Une erreur s'est produite : {str(e)}")
    raise

finally:
    # Nettoyage
    if 'spark' in locals():
        spark.stop()
