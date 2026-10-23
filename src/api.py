from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.validate_readings import inspect_reading
from src.explainability import explain_prediction


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "sentinelx_lightgbm.joblib"
ANOMALY_MODEL_PATH = ROOT / "models" / "sentinelx_anomaly_detector.joblib"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]

anomaly_bundle = joblib.load(ANOMALY_MODEL_PATH)
anomaly_pipeline = anomaly_bundle["pipeline"]
anomaly_features = anomaly_bundle["features"]
anomaly_threshold = float(anomaly_bundle["threshold"])

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

@app.post("/explain")
def explain(reading: EngineReading):
    try:
        result = explain_prediction(
            model,
            features,
            reading.model_dump(),
        )

        return {
            "service": "SentinelX",
            "explanation_type": "SHAP feature attribution",
            "interpretation": (
                "Positive contributions increase predicted RUL; "
                "negative contributions decrease predicted RUL. "
                "These are model attributions, not causal explanations."
            ),
            "predicted_rul_cycles": round(
                max(0.0, result["raw_model_prediction_cycles"]), 2
            ),
            **result,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Explanation failed. Check the server logs.",
        ) from exc

@app.post("/anomaly")
def detect_anomaly(reading: EngineReading):
    try:
        input_data = pd.DataFrame(
            [reading.model_dump()]
        )

        input_data = input_data[anomaly_features]

        score = float(
            anomaly_pipeline.decision_function(input_data)[0]
        )

        flagged = score <= anomaly_threshold

        return {
            "service": "SentinelX",
            "detection_type": "Isolation Forest anomaly detection",
            "anomaly_score": round(score, 8),
            "threshold": round(anomaly_threshold, 8),
            "is_anomaly": bool(flagged),
            "status": (
                "unusual_reading"
                if flagged
                else "normal_looking_reading"
            ),
            "interpretation": (
                "This reading is unusual relative to the "
                "detector's learned reference distribution."
                if flagged
                else
                "This reading was not flagged by the "
                "detector's current threshold."
            ),
            "warning": (
                "An anomaly flag is not a confirmed engine fault. "
                "The detector has not been validated against "
                "ground-truth anomaly labels."
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Anomaly detection failed. Check the server logs.",
        ) from exc
