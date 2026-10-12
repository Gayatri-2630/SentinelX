import numpy as np
import pandas as pd
import shap


def explain_prediction(model, features, reading, top_n=5):
    """
    Explain a single RUL prediction using SHAP.

    SHAP contributions explain the model's output, not physical causation.
    """
    input_data = pd.DataFrame(
        [{feature: reading[feature] for feature in features}],
        columns=features,
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_data)

    raw_prediction = float(model.predict(input_data)[0])
    base_value = float(np.asarray(shap_values.base_values).reshape(-1)[0])
    contributions = np.asarray(shap_values.values)[0]

    feature_contributions = [
        {
            "feature": feature,
            "contribution_cycles": round(float(value), 4),
        }
        for feature, value in zip(features, contributions)
    ]

    # Positive contributions increase predicted RUL; negative ones decrease it.
    feature_contributions.sort(
        key=lambda item: abs(item["contribution_cycles"]),
        reverse=True,
    )

    return {
        "raw_model_prediction_cycles": round(raw_prediction, 2),
        "base_value_cycles": round(base_value, 4),
        "top_features": feature_contributions[:top_n],
        "contributions_sum_cycles": round(float(contributions.sum()), 4),
        "reconstruction_error_cycles": round(
            abs(base_value + float(contributions.sum()) - raw_prediction), 8
        ),
    }
