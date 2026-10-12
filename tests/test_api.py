from fastapi.testclient import TestClient
from src.api import app
from src.validate_readings import inspect_reading
from src.prepare_data import load_training_data

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "SentinelX"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_prediction_endpoint():
    payload = {
        "cycle": 100,
        "setting_1": 0.0,
        "setting_2": 0.0,
        "setting_3": 0.0,
        **{f"sensor_{i}": 0.0 for i in range(1, 22)},
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    result = response.json()
    assert result["predicted_rul_cycles"] >= 0
    assert result["unit"] == "cycles"
    assert result["model"] == "LightGBM"


def test_prediction_rejects_invalid_cycle():
    payload = {
        "cycle": 0,
        "setting_1": 0.0,
        "setting_2": 0.0,
        "setting_3": 0.0,
        **{f"sensor_{i}": 0.0 for i in range(1, 22)},
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_validate_flags_unusual_readings():
    payload = {
        "setting_1": 0.0,
        "setting_2": 0.0,
        "setting_3": 0.0,
        **{f"sensor_{i}": 0.0 for i in range(1, 22)},
    }

    result = inspect_reading(payload)

    assert result["within_observed_ranges"] is False
    assert result["warning_count"] == 22


def test_validate_accepts_observed_readings():
    from src.validate_readings import TRAINING_RANGES

    payload = {
        "cycle": 100,
        "setting_1": 0.0,
        "setting_2": 0.0,
        "setting_3": 100.0,
    }

    for sensor_number in range(1, 22):
        limits = TRAINING_RANGES[f"sensor_{sensor_number}"]
        payload[f"sensor_{sensor_number}"] = (
            limits["min"] + limits["max"]
        ) / 2

    response = client.post("/validate", json=payload)

    assert response.status_code == 200
    result = response.json()
    assert result["within_observed_ranges"] is True
    assert result["warning_count"] == 0

def test_shap_explanation_reconstructs_prediction():
    import joblib
    from src.explainability import explain_prediction

    bundle = joblib.load("models/sentinelx_lightgbm.joblib")
    features = bundle["features"]

    reading = {
        feature: (
            100 if feature == "cycle"
            else 100.0 if feature == "setting_3"
            else 0.0
        )
        for feature in features
    }

    result = explain_prediction(
        bundle["model"],
        features,
        reading,
    )

    assert len(result["top_features"]) == 5
    assert result["reconstruction_error_cycles"] < 1e-6
    assert result["raw_model_prediction_cycles"] > 0

def test_explain_endpoint_returns_shap_attributions():
    reading = {
        "cycle": 100,
        "setting_1": 0.0,
        "setting_2": 0.0,
        "setting_3": 100.0,
    }

    for sensor_number in range(1, 22):
        reading[f"sensor_{sensor_number}"] = 0.0

    response = client.post("/explain", json=reading)

    assert response.status_code == 200

    result = response.json()

    assert result["explanation_type"] == "SHAP feature attribution"
    assert result["predicted_rul_cycles"] >= 0
    assert len(result["top_features"]) == 5
    assert result["reconstruction_error_cycles"] < 1e-6
