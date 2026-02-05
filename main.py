from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import pandas as pd
import joblib
import numpy as np
import os
import mlflow.sklearn  # <--- NUEVO
from datetime import datetime

app = FastAPI(title="API Mercatenerife - Entrega Final")

# --- CONFIGURACIÓN ---
FECHA_REFERENCIA = datetime(2024, 4, 1) 

# Configuración de Databricks para Render
# Asegúrate de que termine en .com (sin / al final)
os.environ["DATABRICKS_HOST"] = "https://dbc-a344da6b-6cd2.cloud.databricks.com"
os.environ["DATABRICKS_TOKEN"] = os.getenv("DATABRICKS_TOKEN", "").strip()

try:
    # --- CAMBIO AQUÍ: Cargamos desde Databricks en lugar de local ---
    model_uri = "models:/mercatenerife_modelo_final/1"
    model = mlflow.sklearn.load_model(model_uri) 
    
    # Las columnas las seguimos cargando de tu archivo pkl local
    model_columns = joblib.load("columnas.pkl")
    
    # Extraemos productos (Validación que ya tenías)
    PRODUCTOS_VALIDOS = [
        col.replace("producto_nombre_", "").replace("producto_", "") 
        for col in model_columns if "producto" in col.lower() and col.lower() != "producto"
    ]
    STATUS = "ok"
except Exception as e:
    print(f"Error al cargar: {e}")
    model = None
    model_columns = None
    PRODUCTOS_VALIDOS = []
    STATUS = "ko"

# --- VALIDACIÓN CON PYDANTIC (Tus validaciones originales mantenidas) ---
class PredictInput(BaseModel):
    mes: int = Field(..., ge=1, le=12)
    dia: int = Field(..., ge=1, le=31)
    año: int = Field(..., ge=2024) 
    producto: str
    es_local: bool

    @validator('producto')
    def producto_debe_existir(cls, v):
        v_upper = v.upper()
        if PRODUCTOS_VALIDOS and v_upper not in [p.upper() for p in PRODUCTOS_VALIDOS]:
            raise ValueError(f"Producto no reconocido. Opciones válidas: {PRODUCTOS_VALIDOS}")
        return v_upper

@app.get("/health")
def health_check():
    return {
        "status": STATUS, 
        "productos_disponibles": PRODUCTOS_VALIDOS,
        "modelo_cargado": model is not None
    }

@app.post("/predict")
def predict(data: PredictInput):
    if not model or not model_columns:
        raise HTTPException(status_code=500, detail="Modelo no cargado correctamente")
    
    try:
        # 1. Creamos el objeto fecha
        fecha_usuario = datetime(data.año, data.mes, data.dia)
        
        # --- NUEVA VALIDACIÓN DE SEGURIDAD ---
        if fecha_usuario < FECHA_REFERENCIA:
            raise HTTPException(
                status_code=400, 
                detail=f"Error: No hay datos históricos antes del {FECHA_REFERENCIA.strftime('%d/%m/%Y')}. Por favor, introduce una fecha posterior."
            )

        # 2. Cálculos de tendencia y tiempo
        dias_tendencia = (fecha_usuario - FECHA_REFERENCIA).days
        dia_semana_calc = fecha_usuario.weekday()

        # 3. Preparación del DataFrame (Ceros iniciales)
        df_input = pd.DataFrame([np.zeros(len(model_columns))], columns=model_columns)
        
        # Rellenamos columnas temporales
        for col in model_columns:
            c_low = col.lower()
            if c_low == 'tendencia': df_input.at[0, col] = dias_tendencia
            elif c_low == 'dia_semana': df_input.at[0, col] = dia_semana_calc
            elif c_low in ['mes', 'month']: df_input.at[0, col] = data.mes
            elif c_low in ['dia', 'day']: df_input.at[0, col] = data.dia
            elif c_low in ['año', 'anio', 'year']: df_input.at[0, col] = data.año

        # 4. Producto
        col_prod_1 = f"producto_nombre_{data.producto}"
        col_prod_2 = f"producto_{data.producto}"
        if col_prod_1 in model_columns: df_input.at[0, col_prod_1] = 1
        elif col_prod_2 in model_columns: df_input.at[0, col_prod_2] = 1

        # 5. Procedencia
        tipo = "LOCAL" if data.es_local else "IMPORTACION"
        col_proc = f"procedencia_{tipo}"
        if col_proc in model_columns: df_input.at[0, col_proc] = 1

        # 6. Predicción final
        prediction = model.predict(df_input[model_columns])
        
        return {
            "fecha_solicitada": fecha_usuario.strftime("%Y-%m-%d"),
            "tendencia_dias": dias_tendencia,
            "producto": data.producto,
            "procedencia": tipo,
            "precio_estimado": round(float(prediction[0]), 2),
            "unidad": "€/kg"
        }
    except HTTPException as he:
        raise he 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el motor de predicción: {str(e)}")
    
