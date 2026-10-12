# SentinelX - Predictive Maintenance for Aircraft Engines

SentinelX is a machine-learning project that estimates the Remaining Useful Life (RUL) of aircraft engines using sensor measurements from NASA's C-MAPSS turbofan engine degradation dataset.

The project includes data preparation, model comparison, official test-set evaluation, sensor-range validation, a trained LightGBM model, a REST API, Docker packaging, and automated tests.

## Key Results

Evaluated on the official FD001 test set:

| Metric | Result |
|---|---:|
| RMSE | 26.08 cycles |
| MAE | 18.95 cycles |

The model was trained using the FD001 training data and evaluated against the provided official test-set RUL labels. These results apply to FD001 and should not be interpreted as a guarantee of performance on other engine configurations.

### Actual vs. Predicted Remaining Useful Life

![Actual vs. predicted RUL](models/fd001_actual_vs_predicted.png)

The plot compares the model's predicted remaining useful life against the
official FD001 test-set labels.

## Tech Stack

- **Language:** Python
- **Machine Learning:** LightGBM, scikit-learn, pandas, NumPy
- **API:** FastAPI, Pydantic, Uvicorn
- **Model persistence:** Joblib
- **Deployment packaging:** Docker
- **Testing:** pytest, FastAPI TestClient

## Project Structure

```text
SentinelX/
|-- models/
|   |-- sentinelx_lightgbm.joblib
|   |-- fd001_actual_vs_predicted.png
|   |-- fd001_test_predictions.csv
|   `-- feature_importance.png
|-- src/
|   |-- api.py
|   |-- prepare_data.py
|   |-- train_final_model.py
|   |-- evaluate_official_test.py
|   |-- analyze_predictions.py
|   |-- validate_readings.py
|   `-- sensor_ranges.json
|-- tests/
|   `-- test_api.py
|-- .dockerignore
|-- .gitignore
|-- Dockerfile
|-- README.md
`-- requirements.txt
```

## Features

- Estimates engine Remaining Useful Life in cycles from sensor readings.
- Provides predictions through a REST API.
- Validates incoming sensor values against observed FD001 training-data ranges.
- Reports out-of-range values as warnings rather than physical safety judgments.
- Packages the model and API in a Docker image.
- Includes automated tests for API behavior and input validation.
- Includes scripts for model comparison, feature analysis, and prediction-error analysis.

## Run Locally

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
pip install -r requirements.txt
```

### 4. Prepare the dataset and model

Download the NASA C-MAPSS dataset separately and place the FD001 files in the location expected by the data-loading scripts. The raw dataset is excluded from this repository because of its size.

The repository includes a trained model at `models/sentinelx_lightgbm.joblib`, so you can start the API without retraining if the model and dependencies are available.

### 5. Start the API

```powershell
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs to explore the interactive API documentation.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Service information |
| GET | `/health` | API and model health |
| POST | `/predict` | Predict engine RUL |
| POST | `/validate` | Validate sensor readings against observed training ranges |

The prediction endpoint expects a positive engine cycle, three operating settings, and 21 sensor values.

### Example Prediction Request

Send a `POST` request to `/predict` with the current engine cycle, three operating settings, and 21 sensor readings.

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

The predicted RUL is an estimate of the remaining useful life in operating cycles, not a certified maintenance decision.


## Run with Docker

Build the image from the project root:

```powershell
docker build -t sentinelx-api .
```

Start the container:

```powershell
docker run --name sentinelx-container -p 8000:8000 sentinelx-api
```

Then visit http://localhost:8000/docs.

## Run Tests

```powershell
python -m pytest .\tests\test_api.py -v
```

## Limitations

- The current trained model targets the FD001 subset of C-MAPSS.
- RUL estimates are model predictions, not maintenance or airworthiness decisions.
- Sensor-range checks reflect observed training data, not certified operating limits.
- Performance on other operating conditions, fault modes, or real-world engines requires separate validation.

## Author

**R. Gayatri**

GitHub: https://github.com/Gayatri-2630
