"""API REST para predecir el riesgo de falla cardíaca.

El modelo servido es el pipeline completo exportado por
``notebooks/2_model_pipeline_cv.ipynb`` (preprocesamiento + clasificador),
de modo que la API recibe las variables clínicas en crudo.

Endpoints
---------
GET  /                 Información general del servicio.
GET  /health           Sonda de vida y disponibilidad (Docker y Kubernetes).
GET  /model-info       Metadatos y métricas del modelo desplegado.
POST /predict          Predicción a partir de una lista ordenada de variables.
POST /predict/patient  Predicción a partir de un objeto con campos nombrados.
"""

import json
import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", APP_DIR / "model.joblib"))
METADATA_PATH = APP_DIR / "model_metadata.json"
THRESHOLD = 0.5

# Orden en que deben llegar las variables en el endpoint /predict.
FEATURES = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
    "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope",
]
NUMERIC = {"Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"}

model = joblib.load(MODEL_PATH)
metadata = (json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            if METADATA_PATH.exists() else {})

app = FastAPI(
    title="Heart Failure Prediction API",
    description=(
        "Servicio de inferencia del proyecto integrador de MLOps. "
        "Devuelve la probabilidad estimada de enfermedad cardíaca "
        "(HeartDisease = 1) y la clase predicha con umbral 0.5."
    ),
    version="1.0.0",
)


class Input(BaseModel):
    """Lista con las 11 variables en el orden de ``FEATURES``."""

    features: list = Field(
        ...,
        description="Valores en el orden: " + ", ".join(FEATURES),
        json_schema_extra={"example": [54, "M", "ASY", 140, 239, 0,
                                       "Normal", 160, "N", 1.2, "Flat"]},
    )


class Patient(BaseModel):
    """Paciente descrito con campos nombrados y validados."""

    Age: int = Field(..., ge=1, le=120, examples=[54])
    Sex: Literal["M", "F"] = Field(..., examples=["M"])
    ChestPainType: Literal["TA", "ATA", "NAP", "ASY"] = Field(
        ..., examples=["ASY"])
    RestingBP: float = Field(..., ge=0, le=300, examples=[140])
    Cholesterol: float = Field(..., ge=0, le=1000, examples=[239])
    FastingBS: Literal[0, 1] = Field(..., examples=[0])
    RestingECG: Literal["Normal", "ST", "LVH"] = Field(
        ..., examples=["Normal"])
    MaxHR: float = Field(..., ge=40, le=250, examples=[160])
    ExerciseAngina: Literal["Y", "N"] = Field(..., examples=["N"])
    Oldpeak: float = Field(..., ge=-5, le=10, examples=[1.2])
    ST_Slope: Literal["Up", "Flat", "Down"] = Field(..., examples=["Flat"])


class Prediction(BaseModel):
    heart_disease_probability: float
    prediction: int


def _to_frame(values: list) -> pd.DataFrame:
    """Convierte la lista ordenada en el DataFrame que espera el pipeline."""
    if len(values) != len(FEATURES):
        raise HTTPException(
            status_code=422,
            detail=(f"Se esperaban {len(FEATURES)} valores en el orden "
                    f"{FEATURES}; se recibieron {len(values)}."),
        )
    row = {}
    for name, value in zip(FEATURES, values):
        if name in NUMERIC:
            try:
                row[name] = float(value)
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"'{name}' debe ser numérico; recibido: {value!r}",
                ) from exc
        else:
            row[name] = str(value)
    return pd.DataFrame([row], columns=FEATURES)


def _predict(X: pd.DataFrame) -> dict:
    proba = float(model.predict_proba(X)[0][1])
    return {"heart_disease_probability": proba,
            "prediction": int(proba > THRESHOLD)}


@app.get("/")
def root():
    return {
        "service": "heart-failure-prediction",
        "docs": "/docs",
        "endpoints": ["/health", "/model-info", "/predict",
                      "/predict/patient"],
        "features_order": FEATURES,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model-info")
def model_info():
    return {
        "estimator": metadata.get("estimator"),
        "metrics": metadata.get("metrics"),
        "trained_at_utc": metadata.get("trained_at_utc"),
        "features": FEATURES,
        "decision_threshold": THRESHOLD,
    }


@app.post("/predict", response_model=Prediction)
def predict(data: Input):
    return _predict(_to_frame(data.features))


@app.post("/predict/patient", response_model=Prediction)
def predict_patient(patient: Patient):
    return _predict(pd.DataFrame([patient.model_dump()], columns=FEATURES))
