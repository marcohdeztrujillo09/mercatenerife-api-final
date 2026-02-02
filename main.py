from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import pandas as pd
import joblib
import numpy as np
import os

app = FastAPI(title="API Mercatenerife - Entrega Final")

# --- CARGA DEL MODELO Y COLUMNAS (MODIFICADO PARA CARGA DIRECTA) ---
try:
    # Cargamos directamente desde la raíz del proyecto
    model = joblib.load("model.pkl")
    model_columns = joblib.load("columnas.pkl")
    
    # Extraemos la lista de productos válidos para la validación
    PRODUCTOS_VALIDOS = [
        col.replace("producto_nombre_", "") 
        for col in model_columns if col.startswith("producto_nombre_")
    ]
    STATUS = "ok"
except Exception as e:
    print(f"Error cargando los archivos: {e}")
    model = None
    model_columns = None
    PRODUCTOS_VALIDOS = []
    STATUS = "ko"

# --- VALIDACIÓN CON PYDANTIC ---
class PredictInput(BaseModel):
    mes: int = Field(..., ge=1, le=12)
    dia: int = Field(..., ge=1, le=31)
    anio: int = Field(..., ge=2024)
    producto: str
    es_local: bool

    @validator('producto')
    def producto_debe_existir(cls, v):
        if v.upper() not in PRODUCTOS_VALIDOS:
            raise ValueError(f"Producto no reconocido. Opciones válidas: {PRODUCTOS_VALIDOS}")
        return v.upper()

@app.get("/health")
def health_check():
    return {
        "status": STATUS, 
        "productos_disponibles": len(PRODUCTOS_VALIDOS),
        "archivos_encontrados": {
            "model": os.path.exists("model.pkl"),
            "columnas": os.path.exists("columnas.pkl")
        }
    }

@app.post("/predict")
def predict(data: PredictInput):
    if not model or not model_columns:
        raise HTTPException(status_code=500, detail="Modelo o columnas no cargados")
    
    try:
        # Creamos DataFrame con ceros
        df_input = pd.DataFrame([np.zeros(len(model_columns))], columns=model_columns)
        
        # 1. Tiempo
        for col in model_columns:
            if col.lower() in ['anio', 'año', 'year']: df_input.at[0, col] = data.anio
            if col.lower() in ['mes', 'month']: df_input.at[0, col] = data.mes
            if col.lower() in ['dia', 'day']: df_input.at[0, col] = data.dia

        # 2. Producto
        col_prod = f"producto_nombre_{data.producto}"
        if col_prod in model_columns:
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