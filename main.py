from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator # Añadimos validator
import mlflow.sklearn
import pandas as pd
import joblib
import numpy as np

app = FastAPI(title="API Mercatenerife - Entrega Final")

# --- CARGA DEL MODELO Y COLUMNAS ---
try:
    with open("run_id.txt", "r") as f:
        run_id = f.read().strip()
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    model_columns = joblib.load("columnas.pkl")
    
    # Extraemos la lista de productos válidos para la validación
    # Buscamos las columnas que empiezan por 'producto_nombre_'
    PRODUCTOS_VALIDOS = [
        col.replace("producto_nombre_", "") 
        for col in model_columns if col.startswith("producto_nombre_")
    ]
    STATUS = "ok"
except Exception as e:
    model = None
    PRODUCTOS_VALIDOS = []
    STATUS = "ko"

# --- VALIDACIÓN CON PYDANTIC ---
class PredictInput(BaseModel):
    mes: int = Field(..., ge=1, le=12)
    dia: int = Field(..., ge=1, le=31)
    anio: int = Field(..., ge=2024)
    producto: str
    es_local: bool

    # Validación extra: El producto DEBE existir en el modelo
    @validator('producto')
    def producto_debe_existir(cls, v):
        if v.upper() not in PRODUCTOS_VALIDOS:
            raise ValueError(f"Producto no reconocido. Opciones válidas: {PRODUCTOS_VALIDOS}")
        return v.upper()

@app.get("/health")
def health_check():
    return {"status": STATUS, "productos_disponibles": len(PRODUCTOS_VALIDOS)}

@app.post("/predict")
def predict(data: PredictInput):
    if not model:
        raise HTTPException(status_code=500, detail="Modelo no cargado")
    
    try:
        df_input = pd.DataFrame([np.zeros(len(model_columns))], columns=model_columns)
        
        # 1. Tiempo
        for col in model_columns:
            if col.lower() in ['anio', 'año', 'year']: df_input.at[0, col] = data.anio
            if col.lower() in ['mes', 'month']: df_input.at[0, col] = data.mes
            if col.lower() in ['dia', 'day']: df_input.at[0, col] = data.dia

        # 2. Producto (Ya validado por Pydantic, así que existe seguro)
        col_prod = f"producto_nombre_{data.producto}"
        df_input.at[0, col_prod] = 1

        # 3. Procedencia
        tipo = "LOCAL" if data.es_local else "IMPORTACION"
        col_proc = f"procedencia_{tipo}"
        if col_proc in model_columns:
            df_input.at[0, col_proc] = 1

        prediction = model.predict(df_input[model_columns])
        
        return {
            "producto": data.producto,
            "precio_estimado": round(float(prediction[0]), 2),
            "unidad": "€/kg"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")