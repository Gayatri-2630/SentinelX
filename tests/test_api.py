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
    df = load_training_data("FD001")
    row = df.iloc[0]

    payload = {
        column: float(row[column])
        for column in (
            ["setting_1", "setting_2", "setting_3"]
            + [f"sensor_{i}" for i in range(1, 22)]
        )
    }

    result = inspect_reading(payload)

    assert result["within_observed_ranges"] is True
    assert result["warning_count"] == 0