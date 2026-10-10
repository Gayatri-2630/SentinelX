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
â”œâ”€â”€ configs/
â”œâ”€â”€ models/
â”‚   â”œâ”€â”€ sentinelx_lightgbm.joblib
â”‚   â”œâ”€â”€ fd001_actual_vs_predicted.png
â”‚   â”œâ”€â”€ fd001_test_predictions.csv
â”‚   â””â”€â”€ feature_importance.png
â”œâ”€â”€ notebooks/
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ api.py
â”‚   â”œâ”€â”€ prepare_data.py
â”‚   â”œâ”€â”€ train_final_model.py
â”‚   â”œâ”€â”€ evaluate_official_test.py
â”‚   â”œâ”€â”€ analyze_predictions.py
â”‚   â”œâ”€â”€ validate_readings.py
â”‚   â””â”€â”€ sensor_ranges.json
â”œâ”€â”€ tests/
â”‚   â””â”€â”€ test_api.py
â”œâ”€â”€ .dockerignore
â”œâ”€â”€ .gitignore
â”œâ”€â”€ Dockerfile
â””â”€â”€ requirements.txt
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

Run the project's data preparation and training scripts as appropriate, or use the included trained model if available.

### 5. Start the API

```powershell
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` to explore the interactive API documentation.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Service information |
| GET | `/health` | API and model health |
| POST | `/predict` | Predict engine RUL |
| POST | `/validate` | Validate sensor readings against observed training ranges |

The prediction endpoint expects a positive engine cycle, three operating settings, and 21 sensor values.

## Run with Docker

Build the image from the project root:

```powershell
docker build -t sentinelx-api .
```

Start the container:

```powershell
docker run --name sentinelx-container -p 8000:8000 sentinelx-api
```

Then visit `http://localhost:8000/docs`.

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

**R.Gayatri**
GitHub: [Gayatri-2630](https://github.com/Gayatri-2630)
