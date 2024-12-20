import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import json
from pyspark.ml import PipelineModel

# Ajouter le répertoire src au PYTHONPATH
app_dir = Path(__file__).parent.absolute()
project_dir = app_dir.parent.parent
sys.path.append(str(project_dir))

from src.utils.spark_utils import create_spark_session
from src.features.feature_engineering import FeatureEngineering

# Configuration de la page
st.set_page_config(
    page_title="Real Estate Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Initialisation de la session Spark
@st.cache_resource
def init_spark():
    """Initialise la session Spark"""
    return create_spark_session("RealEstate_Streamlit")

# Chargement du pipeline et des paramètres
@st.cache_resource
def load_feature_engineering():
    """Charge le pipeline de feature engineering et ses paramètres"""
    spark = init_spark()
    
    # Chemins des fichiers
    models_dir = project_dir / "models"
    pipeline_path = models_dir / "feature_engineering_pipeline"
    params_path = models_dir / "feature_engineering_params.json"
    
    # Charger le pipeline
    pipeline = PipelineModel.load(str(pipeline_path))
    
    # Charger les paramètres
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    # Créer l'instance de FeatureEngineering
    fe = FeatureEngineering(spark)
    fe.fitted_pipeline = pipeline
    fe.capping_values = params['capping_values']
    fe.RARE_CATEGORY_THRESHOLD = params['RARE_CATEGORY_THRESHOLD']
    fe.MISSING_THRESHOLD = params['MISSING_THRESHOLD']
    fe.EIFFEL_COORDS = tuple(params['EIFFEL_COORDS'])
    
    return fe

# Fonction pour créer un DataFrame Spark à partir des inputs
def create_input_df(data_dict):
    """Crée un DataFrame Spark à partir des données d'entrée"""
    spark = init_spark()
    return spark.createDataFrame([data_dict])

# Interface utilisateur
def main():
    st.title("🏠 Prédiction de Prix Immobilier")
    st.write("Entrez les caractéristiques du bien pour obtenir une estimation du prix")
    
    # Chargement des ressources
    spark = init_spark()
    fe = load_feature_engineering()
    
    # Création des colonnes pour le layout
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
    
    # Coordonnées (optionnelles)
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
        
        # Création du DataFrame Spark et application des transformations
        input_df = create_input_df(input_data)
        transformed_df = fe.transform(input_df)
        
        # Affichage des features transformées
        st.subheader("Features transformées")
        # Conversion en pandas pour l'affichage
        pd_df = transformed_df.toPandas()
        st.dataframe(pd_df)
        
        # TODO: Ajouter la prédiction une fois que le modèle sera entraîné
        st.info("Le modèle de prédiction sera bientôt disponible !")

if __name__ == "__main__":
    main()
