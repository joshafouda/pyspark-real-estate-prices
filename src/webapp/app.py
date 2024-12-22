import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import json
from pyspark.ml import PipelineModel
import tempfile
import os

# Ajouter le répertoire src au PYTHONPATH
app_dir = Path(__file__).parent.absolute()
project_dir = app_dir.parent.parent
sys.path.append(str(project_dir))

from src.utils.spark_utils import create_spark_session
from src.features.feature_engineering import FeatureEngineering
from src.models.train import ModelTrainer
from src.features.target_transformer import TargetTransformer

# Configuration de la page
st.set_page_config(
    page_title="Real Estate Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Initialisation de la session Spark et chargement des modèles
@st.cache_resource
def init_resources():
    """Initialise les ressources nécessaires pour l'application."""
    # Créer une session Spark
    spark = create_spark_session("RealEstate_WebApp")
    
    # Charger le modèle et ses dépendances
    rf_model_dir = Path("models/rf")
    model_trainer = ModelTrainer()
    model_trainer.load_model(rf_model_dir)
    
    # Initialiser le feature engineering avec la session Spark
    model_trainer.feature_engineering.spark = spark
    
    # Charger le target transformer
    target_transformer = TargetTransformer.load(str(rf_model_dir / "target_transformer.json"))
    
    return spark, model_trainer.feature_engineering, model_trainer, target_transformer

# Fonction pour créer un DataFrame Spark à partir des inputs
def create_input_df(data_dict):
    """Crée un DataFrame Spark à partir des données d'entrée"""
    spark = init_resources()[0]
    return spark.createDataFrame([data_dict])

def predict_single(input_data, fe, model_trainer, target_transformer):
    """Fait une prédiction pour un bien immobilier unique"""
    # Création du DataFrame et transformation
    input_df = create_input_df(input_data)
    transformed_df = fe.transform(input_df)
    
    # Prédiction (en log)
    predictions = model_trainer.fitted_model.transform(transformed_df)
    
    # Transformation inverse pour obtenir le prix réel
    final_predictions = target_transformer.inverse_transform(predictions, "prediction")
    predicted_price = final_predictions.select("prediction_price").first()[0]
    
    return predicted_price, transformed_df

def predict_batch(input_file, fe, model_trainer, target_transformer):
    """Fait des prédictions pour un lot de biens immobiliers"""
    spark = init_resources()[0]
    
    # Lecture du fichier CSV
    input_df = spark.read.csv(input_file, header=True, inferSchema=True)
    initial_count = input_df.count()
    st.write(f"Nombre de biens immobiliers dans le fichier : {initial_count}")
    
    # Transformation et prédiction
    transformed_df = fe.transform(input_df)
    transformed_count = transformed_df.count()
    st.write(f"Nombre de biens après feature engineering : {transformed_count}")
    
    if transformed_count < initial_count:
        # Afficher les colonnes manquantes
        missing_cols = set(input_df.columns) - set(transformed_df.columns)
        if missing_cols:
            st.warning(f"Colonnes manquantes après transformation : {missing_cols}")
        
        # Afficher les colonnes avec des valeurs nulles
        for col in transformed_df.columns:
            null_count = transformed_df.filter(f"{col} IS NULL").count()
            if null_count > 0:
                st.warning(f"Colonne {col} : {null_count} valeurs nulles")
    
    predictions = model_trainer.fitted_model.transform(transformed_df)
    predictions_count = predictions.count()
    st.write(f"Nombre de prédictions : {predictions_count}")
    
    # Transformation inverse pour obtenir les prix réels
    final_predictions = target_transformer.inverse_transform(predictions, "prediction")
    final_count = final_predictions.count()
    st.write(f"Nombre de prédictions finales : {final_count}")
    
    # Log the schema and show a sample of the final predictions
    st.write("Schema des prédictions finales:")
    final_predictions.printSchema()
    st.write("Exemple de prédictions finales:")
    final_predictions.show(5)
    
    # Sélection des colonnes pertinentes
    result_df = final_predictions.select("id_annonce", "prediction_price")
    
    return result_df

def main():
    st.title("🏠 Prédiction de Prix Immobilier")
    
    # Chargement des ressources
    spark, fe, model_trainer, target_transformer = init_resources()
    
    # Sélection du mode de prédiction
    prediction_mode = st.radio(
        "Mode de prédiction",
        ["Prédiction unique", "Prédiction par lots"]
    )
    
    if prediction_mode == "Prédiction par lots":
        st.subheader("📊 Prédiction par lots")
        uploaded_file = st.file_uploader(
            "Charger un fichier CSV",
            type="csv",
            help="Le fichier doit avoir le même format que X_test.csv"
        )
        
        if uploaded_file is not None:
            # Sauvegarde temporaire du fichier
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Prédiction
                with st.spinner('Prédiction en cours...'):
                    result_df = predict_batch(tmp_path, fe, model_trainer, target_transformer)
                    
                    # Conversion en pandas pour l'affichage
                    pd_results = result_df.toPandas()
                    pd_results = pd_results.rename(columns={'prediction_price': 'prix_predit'})
                    
                    # Formatage des prix pour l'affichage
                    pd_results['prix_predit'] = pd_results['prix_predit'].round(2)
                    
                    # Sauvegarde des résultats dans un fichier
                    pd_results.to_csv('predictions.csv', index=False)
                    st.success("Les prédictions ont été sauvegardées dans 'predictions.csv'")
                    
                    # Affichage des résultats
                    st.subheader("Résultats")
                    st.dataframe(pd_results)
                    
                    # Export des résultats
                    csv = pd_results.to_csv(index=False)
                    st.download_button(
                        "Télécharger les prédictions (CSV)",
                        csv,
                        "predictions.csv",
                        "text/csv",
                        key='download-csv'
                    )
            
            finally:
                # Nettoyage du fichier temporaire
                os.unlink(tmp_path)
    
    else:
        st.subheader("🏡 Prédiction unique")
        
        # Interface de saisie existante
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Caractéristiques principales")
            property_type = st.selectbox(
                "Type de bien",
                ["maison", "appartement"]
            )
            
            size = st.number_input(
                "Surface (m²)",
                min_value=0.0,
                max_value=1000.0,
                value=50.0
            )
            
            nb_rooms = st.number_input(
                "Nombre de pièces",
                min_value=1,
                max_value=20,
                value=2
            )
            
            nb_bedrooms = st.number_input(
                "Nombre de chambres",
                min_value=0,
                max_value=10,
                value=1
            )
            
            floor = st.number_input(
                "Étage",
                min_value=0,
                max_value=20,
                value=0
            )
        
        with col2:
            st.subheader("Caractéristiques secondaires")
            energy_performance = st.selectbox(
                "Performance énergétique",
                ["A", "B", "C", "D", "E", "F", "G", "Unknown"]
            )
            
            ghg_category = st.selectbox(
                "Catégorie GES",
                ["A", "B", "C", "D", "E", "F", "G", "Unknown"]
            )
            
            exposition = st.selectbox(
                "Exposition",
                ["Nord", "Sud", "Est", "Ouest", "Nord-Est", "Nord-Ouest", 
                 "Sud-Est", "Sud-Ouest", "Unknown"]
            )
            
            land_size = st.number_input(
                "Surface du terrain (m²)",
                min_value=0.0,
                max_value=10000.0,
                value=0.0
            )
            
            nb_bathrooms = st.number_input(
                "Nombre de salles de bain",
                min_value=0,
                max_value=5,
                value=1
            )
        
        # Coordonnées
        st.subheader("Localisation")
        col3, col4 = st.columns(2)
        
        with col3:
            latitude = st.number_input(
                "Latitude",
                min_value=41.0,
                max_value=51.0,
                value=48.8566
            )
        
        with col4:
            longitude = st.number_input(
                "Longitude",
                min_value=-5.0,
                max_value=10.0,
                value=2.3522
            )
        
        # Bouton de prédiction
        if st.button("Calculer l'estimation"):
            # Création du dictionnaire d'entrée
            input_data = {
                'property_type': property_type,
                'size': float(size),
                'floor': int(floor),
                'land_size': float(land_size),
                'energy_performance_category': energy_performance,
                'ghg_category': ghg_category,
                'nb_rooms': int(nb_rooms),
                'nb_bedrooms': int(nb_bedrooms),
                'nb_bathrooms': int(nb_bathrooms),
                'exposition': exposition,
                'approximate_latitude': float(latitude),
                'approximate_longitude': float(longitude),
                # Valeurs par défaut pour les autres champs
                'energy_performance_value': None,
                'ghg_value': None,
                'nb_parking_places': 0,
                'nb_boxes': 0,
                'nb_photos': 0
            }
            
            # Prédiction
            with st.spinner('Calcul en cours...'):
                predicted_price, transformed_df = predict_single(input_data, fe, model_trainer, target_transformer)
                
                # Affichage du résultat
                st.subheader("Prix estimé")
                st.metric(
                    "Prix",
                    f"{predicted_price:,.2f} €",
                    help="Cette estimation est basée sur les caractéristiques fournies"
                )
                
                # Affichage des features transformées (optionnel)
                with st.expander("Voir les features transformées"):
                    st.dataframe(transformed_df.toPandas())

if __name__ == "__main__":
    main()
