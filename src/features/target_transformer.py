"""
Module pour la transformation de la variable cible (prix).
"""
from pathlib import Path
import json
import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, log, exp

class TargetTransformer:
    """Classe pour gérer la transformation de la variable cible (prix)."""
    
    def __init__(self, target_col="price"):
        """Initialise le transformateur.
        
        Args:
            target_col: Nom de la colonne cible (default: "price")
        """
        self.target_col = target_col
        self.transformed_col = f"log_{target_col}"
        self.fitted = False
        self.stats = {}
    
    def fit(self, df: DataFrame) -> None:
        """Calcule les statistiques nécessaires sur les données d'entraînement.
        
        Args:
            df: DataFrame Spark contenant la colonne cible
        """
        if self.target_col not in df.columns:
            raise ValueError(f"La colonne {self.target_col} n'existe pas dans le DataFrame")
        
        # Calculer les statistiques sur les données originales
        stats = df.select(self.target_col).summary("count", "mean", "stddev").collect()
        self.stats.update({
            "original_count": float(stats[0][1]),
            "original_mean": float(stats[1][1]),
            "original_std": float(stats[2][1])
        })
        
        # Appliquer la transformation log et calculer les statistiques
        df_log = df.select(log(col(self.target_col)).alias(self.transformed_col))
        stats_log = df_log.summary("mean", "stddev").collect()
        self.stats.update({
            "log_mean": float(stats_log[0][1]),
            "log_std": float(stats_log[1][1])
        })
        
        self.fitted = True
    
    def transform(self, df: DataFrame) -> DataFrame:
        """Applique la transformation logarithmique à la variable cible.
        
        Args:
            df: DataFrame Spark contenant la colonne cible
            
        Returns:
            DataFrame avec la colonne transformée ajoutée
        """
        if not self.fitted:
            raise ValueError("Le transformateur n'a pas été ajusté. Appelez fit() d'abord.")
        
        if self.target_col not in df.columns:
            raise ValueError(f"La colonne {self.target_col} n'existe pas dans le DataFrame")
        
        return df.withColumn(self.transformed_col, log(col(self.target_col)))
    
    def inverse_transform(self, df: DataFrame, prediction_col: str) -> DataFrame:
        """Applique la transformation inverse aux prédictions.
        
        Args:
            df: DataFrame Spark contenant la colonne de prédictions
            prediction_col: Nom de la colonne contenant les prédictions (en log)
            
        Returns:
            DataFrame avec les prédictions transformées en prix réels
        """
        if not self.fitted:
            raise ValueError("Le transformateur n'a pas été ajusté. Appelez fit() d'abord.")
        
        # Correction du biais de transformation logarithmique
        correction = np.exp(self.stats["log_std"]**2 / 2)
        
        return df.withColumn(
            f"{prediction_col}_price",
            exp(col(prediction_col)) * correction
        )
    
    def fit_transform(self, df: DataFrame) -> DataFrame:
        """Ajuste le transformateur et applique la transformation.
        
        Args:
            df: DataFrame Spark contenant la colonne cible
            
        Returns:
            DataFrame avec la colonne transformée ajoutée
        """
        self.fit(df)
        return self.transform(df)
    
    def save(self, path: str) -> None:
        """Sauvegarde les paramètres du transformateur.
        
        Args:
            path: Chemin où sauvegarder les paramètres
        """
        params = {
            "target_col": self.target_col,
            "transformed_col": self.transformed_col,
            "fitted": self.fitted,
            "stats": self.stats
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(params, f, indent=4)
    
    @classmethod
    def load(cls, path: str) -> "TargetTransformer":
        """Charge un transformateur depuis un fichier.
        
        Args:
            path: Chemin vers le fichier de paramètres
            
        Returns:
            Instance de TargetTransformer initialisée avec les paramètres chargés
        """
        with open(path, "r") as f:
            params = json.load(f)
        
        transformer = cls(target_col=params["target_col"])
        transformer.transformed_col = params["transformed_col"]
        transformer.fitted = params["fitted"]
        transformer.stats = params["stats"]
        
        return transformer
