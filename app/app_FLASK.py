import pandas as pd
import pickle
import json
import numpy as np
from flask import Flask 
from flask import request, jsonify
import requests



app = Flask(__name__)

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

# Listado de columnas de cada target
columns_bedrooms = ['A_TOTAL_(m2)', 'N_HABITACIONES', 'N_BAÑOS', 'N_ASEOS', 'TIENE_LAVADERO', 'TIENE_ENTRADA']
columns_bathrooms = ['A_TOTAL_(m2)', 'N_HABITACIONES', 'N_BAÑOS']
columns_livingroom = ['A_TOTAL_(m2)', 'N_HABITACIONES', 'N_ASEOS', 'TIENE_LAVADERO', 'TIENE_ENTRADA']
columns_kitchen = ['A_TOTAL_(m2)', 'N_HABITACIONES', 'N_BAÑOS', 'TIENE_LAVADERO']
columns_toilet = ['A_TOTAL_(m2)', 'N_HABITACIONES', 'N_BAÑOS', 'TIENE_LAVADERO', 'TIENE_ENTRADA']

@app.route('/') 
def home():
    return "Conexión a Modelo de Predicción establecida (DEMO)."  
    
@app.route('/predict', methods=["POST"])
def predict():

    data = request.get_json()

    # Extraer datos con valores por defecto "N/A" para detectar datos faltantes
    total_area = data.get("total_area", "N/A")
    n_bedrooms = data.get("n_bedrooms", "N/A")
    n_bathrooms = data.get("n_bathrooms", "N/A")
    n_toilets = data.get("n_toilets", "N/A")
    laundry = data.get("laundry", "N/A")
    hall = data.get("hall", "N/A")

    # Validación de datos faltantes
    missing_fields = [field for field, value in {
        "total_area": total_area,
        "n_bedrooms": n_bedrooms,
        "n_bathrooms": n_bathrooms,
        "n_toilets": n_toilets,
        "laundry": laundry,
        "hall": hall
    }.items() if value == "N/A"]

    if missing_fields:
        return jsonify({"error": "Faltan datos", "campos_faltantes": missing_fields}), 400
    

    # Validación de tipos de datos y de valores permitidos
    if not isinstance(total_area, (int, float)) or isinstance(total_area, bool):
        return jsonify({"Error": "total_area debe ser un número (int o float), no un booleano"}), 400

    if not isinstance(n_bedrooms, int) or isinstance(n_bedrooms, bool) or n_bedrooms not in range(1,7):
        return jsonify({"Error": "n_bedrooms debe ser un entero entre 1 y 6"}), 400

    if not isinstance(n_bathrooms, int) or isinstance(n_bathrooms, bool) or n_bathrooms not in range(1,4):
        return jsonify({"Error": "n_bathrooms debe ser un entero entre 1 y 3"}), 400

    if not isinstance(n_toilets, int) or isinstance(n_toilets, bool) or n_toilets not in range(0,2):
        return jsonify({"Error": "n_toilets solo puede contener '0' o '1'"}), 400 

    if not isinstance(laundry, bool):
        return jsonify({"Error": "laundry debe ser un booleano"}), 400

    if not isinstance(hall, bool):
        return jsonify({"Error": "hall debe ser un booleano"}), 400

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
    
    # Calcular áreas individuales usando proporciones
    room_details = {}

    # Habitaciones
    room_details["bedrooms"] = {"area": pred_bedrooms, "details": {}}
    if n_bedrooms > 1:
        proportions = bedrooms_prop[f"prop_{n_bedrooms}_rooms"]
        for i, proportion in enumerate(proportions):
            room_size = round(pred_bedrooms * proportion, 2)
            room_details["bedrooms"]["details"][f"bedroom_{i + 1}"] = room_size
    else:
        room_details["bedrooms"]["details"]["bedroom_1"] = pred_bedrooms


    # 🔹 Baños
    room_details["bathrooms"] = {"area": pred_bathrooms, "details": {}}
    if n_bathrooms > 1:
        proportions = bathrooms_prop[f"prop_{n_bathrooms}_bathrooms"]
        for i, proportion in enumerate(proportions):
            bathroom_size = round(pred_bathrooms * proportion, 2)
            room_details["bathrooms"]["details"][f"bathroom_{i + 1}"] = bathroom_size
    
    else:

         room_details["bathrooms"]["details"]["bathroom_1"] = pred_bathrooms


    # 🔹 Otras áreas
    room_details["livingroom"] = {
        "area": pred_livingroom
    }
    room_details["kitchen"] = {
        "area": pred_kitchen
    }
    if n_toilets == 1:
        room_details["toilet"] = {
            "area": pred_toilet
        }

    return jsonify(room_details), 200


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
