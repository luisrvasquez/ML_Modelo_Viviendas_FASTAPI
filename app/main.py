import pandas as pd
import pickle
import json
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Importar esquemas
from app.models.schemas import PropertyCharacteristics
from app.models.schemas import PropertySpaces


app = FastAPI()

# Seguridad CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modificar en producción 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_pickle(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)

# Cargar modelos entrenados
linear_model_bedrooms = load_pickle("app/modelos/linear_model_bedrooms.pkl")
linear_model_bathrooms = load_pickle("app/modelos/linear_model_bathrooms.pkl")
linear_model_livingroom = load_pickle("app/modelos/linear_model_livingroom.pkl")
linear_model_kitchen = load_pickle("app/modelos/linear_model_kitchen.pkl")
rf_model_toilet = load_pickle("app/modelos/rf_model_toilet.pkl")

# Cargar escaladores
scaler_bedrooms = load_pickle("app/modelos/scaler_bedrooms.pkl")
scaler_bathrooms = load_pickle("app/modelos/scaler_bathrooms.pkl")
scaler_livingroom = load_pickle("app/modelos/scaler_livingroom.pkl")
scaler_kitchen = load_pickle("app/modelos/scaler_kitchen.pkl")
scaler_toilet = load_pickle("app/modelos/scaler_toilet.pkl")

# Cargar métricas y proporciones
with open("app/data/room_metrics.json", "r") as file:
    rooms_metrics = json.load(file)

with open("app/data/models_mae.json", "r") as file:
    mae_metrics = json.load(file)

with open("app/data/bedrooms_prop.json", "r") as file:
    bedrooms_prop = json.load(file)

with open("app/data/bathrooms_prop.json", "r") as file:
    bathrooms_prop = json.load(file)


# Se cargan los maes
bedrooms_mae = mae_metrics["bedrooms_mae"]
bathrooms_mae = mae_metrics["bathrooms_mae"]
livingroom_mae = mae_metrics["livingroom_mae"]
kitchen_mae = mae_metrics["kitchen_mae"]
toilet_mae = mae_metrics["toilet_mae"]

# Se cargan las proporciones
prop_2_rooms = bedrooms_prop["prop_2_rooms"]
prop_3_rooms = bedrooms_prop["prop_3_rooms"]
prop_4_rooms = bedrooms_prop["prop_4_rooms"]
prop_5_rooms = bedrooms_prop["prop_5_rooms"]
prop_6_rooms = bedrooms_prop["prop_6_rooms"]

prop_2_bathrooms = bathrooms_prop["prop_2_bathrooms"]
prop_3_bathrooms = bathrooms_prop["prop_3_bathrooms"]

# Listado de columnas para cada modelo
columns_bedrooms = ['A_TOTAL_(m2)', 'N_HABITACIONES', 'N_BAÑOS', 'N_ASEOS', 'TIENE_LAVADERO', 'TIENE_ENTRADA']
columns_bathrooms = ['A_TOTAL_(m2)', 'N_HABITACIONES', 'N_BAÑOS']
columns_livingroom = ['A_TOTAL_(m2)', 'N_HABITACIONES', 'N_ASEOS', 'TIENE_LAVADERO', 'TIENE_ENTRADA']
columns_kitchen = ['A_TOTAL_(m2)', 'N_HABITACIONES', 'N_BAÑOS', 'TIENE_LAVADERO']
columns_toilet = ['A_TOTAL_(m2)', 'N_HABITACIONES', 'N_BAÑOS', 'TIENE_LAVADERO', 'TIENE_ENTRADA']


@app.get("/")
def home():
    return {"message": "Connection established."}


@app.post("/predict",response_model=PropertySpaces)
async def predict(data: PropertyCharacteristics):

    # Validación de inputs
    if any(v is None for v in [data.built_sqm, data.usable_sqm, data.number_of_bedrooms, data.number_of_bathrooms, data.number_of_toilets]):
        return JSONResponse(
        status_code=400,
        content={"error": "Required fields: 'built_sqm', 'usable_sqm', 'number_of_bedrooms', 'number_of_bathrooms', 'number_of_toilets'"}
        )


    # Extracción de campos esperados desde el JSON original
    
    # Gestión de área útil y de área construida

    if data.usable_sqm > 0:
        total_area = data.usable_sqm
    else:
        total_area = data.built_sqm

    # Gestión de valores enteros
    n_bedrooms = int(data.number_of_bedrooms)
    n_bathrooms = int(data.number_of_bathrooms)
    n_toilets = int(data.number_of_toilets)

    # Tratamiento de Hall y Laundry

    # Laundry
    if data.laundry is None:
        laundry = False
    else:
        laundry = data.laundry

    # Hall
    if data.entrance_hall is None:
        hall = False
    else:
        hall = data.entrance_hall



    # Generación de inputs para predicción
    input_bedrooms = scaler_bedrooms.transform(pd.DataFrame([[total_area, n_bedrooms, n_bathrooms, n_toilets, laundry, hall]], columns=columns_bedrooms))
    input_bathrooms = scaler_bathrooms.transform(pd.DataFrame([[total_area, n_bedrooms, n_bathrooms]], columns=columns_bathrooms))
    input_livingroom = scaler_livingroom.transform(pd.DataFrame([[total_area, n_bedrooms, n_toilets, laundry, hall]], columns=columns_livingroom))
    input_kitchen = scaler_kitchen.transform(pd.DataFrame([[total_area, n_bedrooms, n_bathrooms, laundry]], columns=columns_kitchen))
    input_toilet = scaler_toilet.transform(pd.DataFrame([[total_area, n_bedrooms, n_bathrooms, laundry, hall]], columns=columns_toilet))

    # Predicciones
    pred_bedrooms = round(linear_model_bedrooms.predict(input_bedrooms)[0][0], 2)
    pred_bathrooms = round(linear_model_bathrooms.predict(input_bathrooms)[0][0], 2)
    pred_livingroom = round(linear_model_livingroom.predict(input_livingroom)[0][0], 2)
    pred_kitchen = round(linear_model_kitchen.predict(input_kitchen)[0][0], 2)
    pred_toilet = round(rf_model_toilet.predict(input_toilet)[0], 2)

    # Construcción de respuesta estructurada
    bedrooms_list = []
    if n_bedrooms > 1:
        proportions = bedrooms_prop[f"prop_{n_bedrooms}_rooms"]
        for proportion in proportions:
            room_size = round(pred_bedrooms * proportion, 2)
            bedrooms_list.append({"area": room_size, "perimeter": 0.0})

    elif n_bedrooms == 1:
        bedrooms_list.append({"area": pred_bedrooms, "perimeter": 0.0})
    
    else:
        bedrooms_list.append({"area": 0.0, "perimeter": 0.0})

    bathrooms_list = []
    if n_bathrooms > 1:
        proportions = bathrooms_prop[f"prop_{n_bathrooms}_bathrooms"]
        for proportion in proportions:
            bathroom_size = round(pred_bathrooms * proportion, 2)
            bathrooms_list.append({"area": bathroom_size, "perimeter": 0.0})

    elif n_bathrooms == 1:
        bathrooms_list.append({"area": pred_bathrooms, "perimeter": 0.0})
    
    else:
        bathrooms_list.append({"area": 0.0, "perimeter": 0.0})

    toilets_list = []
    if n_toilets > 0:
        for toilet in  range(0, n_toilets):
            toilets_list.append({"area": pred_toilet, "perimeter": 0.0})
    else:
        toilets_list.append({"area": 0.0, "perimeter": 0.0})

    response = {
        "kitchen": {"area": pred_kitchen, "perimeter": 0.0},
        "livingRoom": {"area": pred_livingroom, "perimeter": 0.0},
        "laundry": None,
        "lounge": {"area": 0.0, "perimeter": 0.0},
        "hall": {"area": 0.0, "perimeter": 0.0},
        "dressing": None,
        "corridors": {"area": 0.0, "perimeter": 0.0},
        "other": {"area": 0.0, "perimeter": 0.0},
        "bedrooms": bedrooms_list,
        "bathrooms": bathrooms_list,
        "toilets": toilets_list,
        "nStairs": 0.0,
        "nSteps": 0.0,
        "nRooms": float(n_bedrooms + n_bathrooms + n_toilets + 2)
    }

    return JSONResponse(content=response, status_code=200)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
