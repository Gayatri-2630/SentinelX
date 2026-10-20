# SentinelX — Predictive Maintenance for Aircraft Engines

SentinelX is a machine learning project that estimates the **Remaining Useful Life (RUL)** of aircraft engines using sensor measurements from NASA's C-MAPSS turbofan engine degradation dataset.

The project combines a trained LightGBM regression model, a REST API, sensor-range validation, SHAP-based prediction explainability, Docker packaging, and automated tests.

## 🚀 Live Demo

- **Live API:** https://sentinelx-o4im.onrender.com
- **Interactive API Documentation:** https://sentinelx-o4im.onrender.com/docs
- **Health Check:** https://sentinelx-o4im.onrender.com/health

Try the deployed API to generate Remaining Useful Life (RUL) predictions, inspect SHAP-based feature explanations, and validate sensor readings against observed FD001 training-data ranges.

> **Note:** SentinelX is a research and portfolio project, not a certified aircraft safety system. The free hosting instance may take around 50 seconds or more to respond after periods of inactivity.

## Key Results

Evaluated on the official NASA C-MAPSS FD001 test set.

| Metric | Result |
|---|---:|
| Root Mean Squared Error (RMSE) | 26.08 cycles |
| Mean Absolute Error (MAE) | 18.95 cycles |

The model was trained on FD001 training data and evaluated against the provided official test-set RUL labels. These metrics describe performance on FD001 and do not guarantee similar results on other engine configurations or real-world engines.

### Actual vs. Predicted Remaining Useful Life

![Actual vs. predicted RUL](models/fd001_actual_vs_predicted.png)

The plot compares the model's predicted remaining useful life against the official FD001 test-set labels.

## Features

- **RUL prediction:** Estimates remaining useful life in operating cycles using LightGBM.
- **REST API:** Provides prediction, validation, and explanation endpoints through FastAPI.
- **SHAP explainability:** Identifies features contributing most strongly to an individual model prediction.
- **Sensor-range validation:** Flags readings outside the observed FD001 training-data ranges.
- **Model evaluation:** Includes official test-set evaluation and prediction-analysis scripts.
- **Docker support:** Packages the trained model and API into a container.
- **Automated testing:** Tests API behavior, input validation, and SHAP explainability.

## Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Machine learning | LightGBM, scikit-learn, SHAP |
| Data processing | pandas, NumPy |
| API | FastAPI, Pydantic, Uvicorn |
| Model persistence | Joblib |
| Containerization | Docker |
| Testing | pytest, FastAPI TestClient |

## Project Structure

```text
SentinelX/
├── models/
│   ├── sentinelx_lightgbm.joblib
│   ├── fd001_actual_vs_predicted.png
│   ├── fd001_test_predictions.csv
│   └── feature_importance.png
├── src/
│   ├── api.py
│   ├── explainability.py
│   ├── prepare_data.py
│   ├── train_final_model.py
│   ├── evaluate_official_test.py
│   ├── analyze_predictions.py
│   ├── validate_readings.py
│   └── sensor_ranges.json
├── tests/
│   └── test_api.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.12
- Git
- Docker Desktop (optional, for containerized execution)

### 1. Clone the repository

```bash
git clone https://github.com/Gayatri-2630/SentinelX.git
cd SentinelX
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Start the API

The repository includes a trained model at `models/sentinelx_lightgbm.joblib`. You do **not** need to download the NASA dataset or retrain the model to run the API.

```powershell
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open the interactive API documentation:

**http://127.0.0.1:8000/docs**

The NASA C-MAPSS FD001 dataset is needed only if you want to rerun data preparation, model training, or official test-set evaluation. The raw dataset is excluded from this repository because of its size.

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Returns service information |
| GET | `/health` | Reports API health and whether the model is loaded |
| POST | `/predict` | Predicts remaining useful life |
| POST | `/validate` | Checks readings against observed FD001 training ranges |
| POST | `/explain` | Returns an RUL prediction and SHAP feature attributions |

The prediction and explanation endpoints expect an engine cycle, three operating settings, and 21 sensor readings.

### Example Prediction Request

Send a `POST` request to `/predict` with a JSON body containing the current engine cycle, operating settings, and sensor measurements.

```json
{
  "cycle": 100,
  "setting_1": -0.0021,
  "setting_2": -0.0003,
  "setting_3": 100.0,
  "sensor_1": 518.67,
  "sensor_2": 642.22,
  "sensor_3": 1589.63,
  "sensor_4": 1411.35,
  "sensor_5": 14.62,
  "sensor_6": 21.61,
  "sensor_7": 554.22,
  "sensor_8": 2388.07,
  "sensor_9": 9045.72,
  "sensor_10": 1.3,
  "sensor_11": 47.64,
  "sensor_12": 521.55,
  "sensor_13": 2388.08,
  "sensor_14": 8132.49,
  "sensor_15": 8.4241,
  "sensor_16": 0.03,
  "sensor_17": 392.0,
  "sensor_18": 2388.0,
  "sensor_19": 100.0,
  "sensor_20": 38.93,
  "sensor_21": 23.409
}
```

Example response:

```json
{
  "predicted_rul_cycles": 109.83,
  "unit": "cycles",
  "model": "LightGBM"
}
```

The response above is the example previously documented for this input. Predictions are estimates, not certified maintenance or airworthiness decisions.

### Sensor Validation

Send the same input structure to `POST /validate`.

The endpoint checks each reading against the observed minimum and maximum values in the FD001 training data and returns warnings for values outside those ranges.

These checks are **data-range checks, not physical safety limits**. A reading inside the observed range is not proof that an engine is healthy, and a reading outside it is not by itself proof of a fault.

### SHAP Explainability

Send a `POST` request to `/explain` using the same JSON payload shown in the prediction example.

The response includes:

- `predicted_rul_cycles`: the non-negative RUL value returned by the endpoint.
- `raw_model_prediction_cycles`: the model's unmodified prediction.
- `base_value_cycles`: the SHAP reference value.
- `top_features`: the five features with the largest absolute SHAP contributions.
- `contributions_sum_cycles`: the sum of all feature contributions.
- `reconstruction_error_cycles`: the absolute difference between the reconstructed and raw model prediction.

Positive SHAP contributions push the model prediction higher relative to the reference value; negative contributions push it lower. These values describe model attribution, not physical causation.

The following is an **illustrative response structure**, not a verified result for the example prediction input. The numerical values are placeholders.

```json
{
  "service": "SentinelX",
  "explanation_type": "SHAP feature attribution",
  "interpretation": "Positive contributions increase predicted RUL; negative contributions decrease predicted RUL. These are model attributions, not causal explanations.",
  "predicted_rul_cycles": 0.0,
  "raw_model_prediction_cycles": 0.0,
  "base_value_cycles": 0.0,
  "top_features": [
    {
      "feature": "sensor_11",
      "contribution_cycles": 0.0
    }
  ],
  "contributions_sum_cycles": 0.0,
  "reconstruction_error_cycles": 0.0
}
```

The actual endpoint returns five top features and includes contributions from all 25 model input features in its reconstruction calculation. Small floating-point differences may occur.

## Run with Docker

### 1. Build the image

From the project root:

```powershell
docker build -t sentinelx:local .
```

### 2. Start the container

Run the container in detached mode:

```powershell
docker run -d --name sentinelx-test -p 10000:10000 sentinelx:local
```

### 3. Check the service health

```powershell
Invoke-RestMethod http://localhost:10000/health
```

The service should report a healthy status and confirm that the model is loaded.

### 4. Open the API documentation

Visit **http://localhost:10000/docs** to explore the API endpoints interactively.

### 5. Stop and remove the container

```powershell
docker stop sentinelx-test
docker rm sentinelx-test
```

## Run Tests

Activate the virtual environment and run:

```powershell
python -m pytest -q
```

The test suite covers the health and home endpoints, prediction behavior, invalid input handling, sensor-range validation, and SHAP explanation behavior.

## Limitations

- The current model targets the FD001 subset of the NASA C-MAPSS dataset.
- RUL predictions are estimates and should not be used as standalone maintenance or airworthiness decisions.
- Sensor validation checks observed training-data ranges rather than certified operating limits.
- SHAP values explain model behavior; they do not establish causal relationships.
- Performance on other engine configurations, fault modes, and real-world engines requires separate evaluation.
- The API expects all three operating settings and all 21 sensor values for each request.

## Author

**R. Gayatri**

- GitHub: https://github.com/Gayatri-2630
- Project repository: https://github.com/Gayatri-2630/SentinelX
