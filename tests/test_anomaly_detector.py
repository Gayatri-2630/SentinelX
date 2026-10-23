
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "sentinelx_anomaly_detector.joblib"


def load_detector():
    return joblib.load(MODEL_PATH)


def test_anomaly_model_bundle_has_expected_structure():
    bundle = load_detector()

    assert bundle["algorithm"] == "IsolationForest"
    assert bundle["dataset"] == "FD001"
    assert len(bundle["features"]) == 24
    assert "scaler" in bundle["pipeline"].named_steps
    assert "detector" in bundle["pipeline"].named_steps


def test_anomaly_detector_returns_valid_predictions_and_scores():
    bundle = load_detector()
    features = bundle["features"]

    # Synthetic input for a technical smoke test, not a real engine reading.
    sample = pd.DataFrame(
        [np.zeros(len(features))],
        columns=features,
    )

    predictions = bundle["pipeline"].predict(sample)
    scores = bundle["pipeline"].decision_function(sample)

    assert predictions.shape == (1,)
    assert predictions[0] in (-1, 1)
    assert scores.shape == (1,)
    assert np.isfinite(scores[0])


def test_anomaly_detector_accepts_multiple_readings():
    bundle = load_detector()
    features = bundle["features"]

    samples = pd.DataFrame(
        np.zeros((5, len(features))),
        columns=features,
    )

    predictions = bundle["pipeline"].predict(samples)
    scores = bundle["pipeline"].decision_function(samples)

    assert predictions.shape == (5,)
    assert set(predictions).issubset({-1, 1})
    assert np.isfinite(scores).all()


def test_anomaly_detector_uses_expected_feature_names():
    bundle = load_detector()

    expected_features = (
        ["setting_1", "setting_2", "setting_3"]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    assert bundle["features"] == expected_features
