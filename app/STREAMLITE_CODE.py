import streamlit as st
import pandas as pd
import pickle
import json
import numpy as np

# Estilos personalizados
st.markdown(
    """
    <style>
    /* Fondo y estilo general de la aplicación */
    .stApp {
        background-color: #E8F1FA; /* Fondo claro */
        color: #2C2C2C; /* Texto gris oscuro */
        font-family: Arial, sans-serif; /* Fuente limpia */
    }

    /* Fondo y estilo del sidebar */
    [data-testid="stSidebar"] {
        background-color: #BFD8F0  !important; /* Fondo azul oscuro del sidebar */
        color: #2C2C2C; /* Texto gris oscuro */
    
    /* Botones personalizados */
    .stButton > button {
        background-color: #0057B7; /* Azul corporativo */
        color: white !important; /* Texto blanco forzado */
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
    }
    .stButton > button:hover {
        background-color: #003366; /* Azul más oscuro al pasar el cursor */
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Controlar distribucion del contenido
st.markdown(
    """
    <style>
    .block-container {
        max-width: 90%; /* Cambiar el ancho máximo del contenido */
        padding-left: 5%;
        padding-right: 5%;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Cargar modelos y datos
def load_pickle(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)

# Cargar modelos entrenados
linear_model_bedrooms = load_pickle("./modelos/linear_model_bedrooms.pkl")
linear_model_bathrooms = load_pickle("./modelos/linear_model_bathrooms.pkl")
linear_model_livingroom = load_pickle("./modelos/linear_model_livingroom.pkl")
linear_model_kitchen = load_pickle("./modelos/linear_model_kitchen.pkl")
rf_model_toilet = load_pickle("./modelos/rf_model_toilet.pkl")

# Cargar escaladores
scaler_bedrooms = load_pickle("./modelos/scaler_bedrooms.pkl")
scaler_bathrooms = load_pickle("./modelos/scaler_bathrooms.pkl")
scaler_livingroom = load_pickle("./modelos/scaler_livingroom.pkl")
scaler_kitchen = load_pickle("./modelos/scaler_kitchen.pkl")
scaler_toilet = load_pickle("./modelos/scaler_toilet.pkl")

# Cargar métricas y proporciones
with open("./data/room_metrics.json", "r") as file:
    rooms_metrics = json.load(file)

with open("./data/models_mae.json", "r") as file:
    mae_metrics = json.load(file)

with open("./data/bedrooms_prop.json", "r") as file:
    bedrooms_prop = json.load(file)

with open("./data/bathrooms_prop.json", "r") as file:
    bathrooms_prop = json.load(file)

# Se cargan los maes
bedrooms_mae = mae_metrics["bedrooms_mae"]
bathrooms_mae = mae_metrics["bathrooms_mae"]
livingroom_mae = mae_metrics["livingroom_mae"]
kitchen_mae = mae_metrics["kitchen_mae"]
toilet_mae = mae_metrics["toilet_mae"]

# Se cargan las proporciones de los dormitorios
prop_2_rooms = bedrooms_prop["prop_2_rooms"]
prop_3_rooms = bedrooms_prop["prop_3_rooms"]
prop_4_rooms = bedrooms_prop["prop_4_rooms"]
prop_5_rooms = bedrooms_prop["prop_5_rooms"]
prop_6_rooms = bedrooms_prop["prop_6_rooms"]

# Se cargan las proporciones de los baños
prop_2_bathrooms = bathrooms_prop["prop_2_bathrooms"]
prop_3_bathrooms = bathrooms_prop["prop_3_bathrooms"]

# Función para realizar predicciones
def make_prediction(total_area, n_bedrooms, n_bathrooms, n_toilets, laundry, hall):
    # Listado de columnas de cada target
    columns_bedrooms = ['A_TOTAL_(m2)', 'N_HABITACIONES', 'N_BAÑOS', 'N_ASEOS', 'TIENE_LAVADERO', 'TIENE_ENTRADA']
    columns_bathrooms = ['A_TOTAL_(m2)', 'N_HABITACIONES', 'N_BAÑOS']
    columns_livingroom = ['A_TOTAL_(m2)', 'N_HABITACIONES', 'N_ASEOS', 'TIENE_LAVADERO', 'TIENE_ENTRADA']
    columns_kitchen = ['A_TOTAL_(m2)', 'N_HABITACIONES', 'N_BAÑOS', 'TIENE_LAVADERO']
    columns_toilet = ['A_TOTAL_(m2)', 'N_HABITACIONES', 'N_BAÑOS', 'TIENE_LAVADERO', 'TIENE_ENTRADA']

    # Generación de Inputs
    input_bedrooms = scaler_bedrooms.transform(pd.DataFrame([[total_area, n_bedrooms, n_bathrooms, n_toilets, laundry, hall]],columns=columns_bedrooms))
    input_bathrooms = scaler_bathrooms.transform(pd.DataFrame([[total_area, n_bedrooms, n_bathrooms]],columns=columns_bathrooms))
    input_livingroom = scaler_livingroom.transform(pd.DataFrame([[total_area, n_bedrooms, n_toilets, laundry, hall]],columns=columns_livingroom))
    input_kitchen = scaler_kitchen.transform(pd.DataFrame([[total_area, n_bedrooms, n_bathrooms, laundry]],columns=columns_kitchen))
    input_toilet = scaler_toilet.transform(pd.DataFrame([[total_area, n_bedrooms, n_bathrooms, laundry, hall]],columns=columns_toilet))
    
    # Generación de Predicciones 
    pred_bedrooms = round(linear_model_bedrooms.predict(input_bedrooms)[0][0], 2)
    pred_bathrooms = round(linear_model_bathrooms.predict(input_bathrooms)[0][0], 2)
    pred_livingroom = round(linear_model_livingroom.predict(input_livingroom)[0][0], 2)
    pred_kitchen = round(linear_model_kitchen.predict(input_kitchen)[0][0], 2)
    pred_toilet = round(rf_model_toilet.predict(input_toilet)[0], 2)
    
    return pred_bedrooms, pred_bathrooms, pred_livingroom, pred_kitchen, pred_toilet


# Título de la aplicación
# Dividir el espacio en columnas
col1, col2 = st.columns([1, 2])  # Ajusta el ancho relativo de las columnas

with col2:
    st.title("Home Spaces Forecaster")
    st.subheader("Predicción profesional de espacios habitacionales.")
    st.write("¿Necesitas una proyección del tamaño de las diferentes áreas de una casa? Introduce los datos y haremos las mediciones.")

with col1:
    st.image("./imagenes/arquitecto.png", caption="A retos exigentes, soluciones inteligentes.",width=400)

with col2:
    # Entradas de usuario
    st.sidebar.header("Datos de la Casa")
    total_area = st.sidebar.number_input("Área Total (m²):", min_value=0.0, step=0.1)
    n_bedrooms = st.sidebar.slider("Número de Dormitorios:", 1, 6, 2)
    n_bathrooms = st.sidebar.slider("Número de Baños:", 1, 3, 1)
    n_toilets = st.sidebar.slider("Número de Aseos:", 0, 1, 0)
    laundry = st.sidebar.selectbox("¿Tiene Lavadero?", ["No", "Sí"]) == "Sí"
    hall = st.sidebar.selectbox("¿Tiene Entrada / Hall?", ["No", "Sí"]) == "Sí"

    # Botón para generar predicciones
    if st.sidebar.button("Generar Predicciones"):
        pred_bedrooms, pred_bathrooms, pred_livingroom, pred_kitchen, pred_toilet = make_prediction(
            total_area, n_bedrooms, n_bathrooms, n_toilets, laundry, hall
        )

        # Mostrar resultados
        st.subheader("Resultados de Predicciones")

        # Dormitorios
        st.write("### Área de Dormitorios")
        bedrooms_lower = pred_bedrooms - bedrooms_mae
        bedrooms_upper = pred_bedrooms + bedrooms_mae
        
        if n_bedrooms == 1:
            mae_bedroom_1 = rooms_metrics['mae_bedroom_1']
            st.write(f"Tamaño del Dormitorio Único: {pred_bedrooms} m² (Rango: {round(bedrooms_lower, 2)} - {round(bedrooms_upper, 2)} m²)")
        else:
            st.write(f"**Área Total de Dormitorios:** {pred_bedrooms} m² (Rango estimado: {round(bedrooms_lower, 2)} - {round(bedrooms_upper, 2)} m²)")
            proportions = eval(f"prop_{n_bedrooms}_rooms")  # Acceso dinámico a las proporciones
            for i, proportion in enumerate(proportions):
                mae_key = f'mae_bedroom_{i + 1}'
                mae = rooms_metrics[mae_key]
                room_size = round(pred_bedrooms * proportion, 2)
                st.write(f"Tamaño del Dormitorio {i + 1}: {room_size} m² (Rango: {round(room_size - mae, 2)} - {round(room_size + mae, 2)} m²)")

        # Baños
        st.write("### Área de Baños")
        bathrooms_lower = pred_bathrooms - bathrooms_mae
        bathrooms_upper = pred_bathrooms + bathrooms_mae

        if n_bathrooms == 1:
            mae_bathroom_1 = rooms_metrics['mae_bathroom_1']
            st.write(f"Tamaño del Baño Único: {pred_bathrooms} m² (Rango: {round(bathrooms_lower, 2)} - {round(bathrooms_upper, 2)} m²)")
        else:
            st.write(f"**Área Total de Baños:** {pred_bathrooms} m² (Rango estimado: {round(bathrooms_lower, 2)} - {round(bathrooms_upper, 2)} m²)")
            proportions = eval(f"prop_{n_bathrooms}_bathrooms")  # Acceso dinámico a las proporciones
            for i, proportion in enumerate(proportions):
                mae_key = f'mae_bathroom_{i + 1}'
                mae = rooms_metrics[mae_key]
                bathroom_size = round(pred_bathrooms * proportion, 2)
                st.write(f"Tamaño del Baño {i + 1}: {bathroom_size} m² (Rango: {round(bathroom_size - mae, 2)} - {round(bathroom_size + mae, 2)} m²)")

        # Salón-Comedor
        st.write("### Área del Salón-Comedor")
        st.write(f"**Tamaño del Salón-Comedor:** {pred_livingroom} m² (Rango estimado: {round(pred_livingroom - livingroom_mae, 2)} - {round(pred_livingroom + livingroom_mae, 2)} m²)")

        # Cocina
        st.write("### Área de la Cocina")
        st.write(f"**Tamaño de la Cocina:** {pred_kitchen} m² (Rango estimado: {round(pred_kitchen - kitchen_mae, 2)} - {round(pred_kitchen + kitchen_mae, 2)} m²)")

        # Aseo (si aplica)
        if n_toilets == 1:
            st.write("### Área del Aseo")
            st.write(f"**Tamaño del Aseo:** {pred_toilet} m² (Rango estimado: {round(pred_toilet - toilet_mae, 2)} - {round(pred_toilet + toilet_mae, 2)} m²)")

