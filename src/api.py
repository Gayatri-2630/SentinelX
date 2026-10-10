from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.validate_readings import inspect_reading


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "sentinelx_lightgbm.joblib"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]

app = FastAPI(
    title="SentinelX API",
    description="Predictive maintenance and remaining useful life estimation",
    version="1.0.0",
)


class EngineReading(BaseModel):
    cycle: int = Field(gt=0, description="Current operating cycle")

    setting_1: float
    setting_2: float
    setting_3: float

    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float


@app.get("/")
def home():
    return {
        "service": "SentinelX",
        "status": "running",
        "model": "LightGBM",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(reading: EngineReading):
    try:
        input_data = pd.DataFrame(
            [reading.model_dump()]
        )

        input_data = input_data[features]

        prediction = float(model.predict(input_data)[0])

        return {
            "predicted_rul_cycles": round(max(0.0, prediction), 2),
            "unit": "cycles",
            "model": "LightGBM",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the server logs.",
        ) from exc

@app.post("/validate")
def validate(reading: EngineReading):
    result = inspect_reading(reading.model_dump())

    return {
        "service": "SentinelX",
        "dataset_reference": "FD001 training ranges",
        **result,
    }
