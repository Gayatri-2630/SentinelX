from pathlib import Path
import joblib

from lightgbm import LGBMRegressor
from prepare_data import load_training_data


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = load_training_data("FD001")

features = [
    col for col in df.columns
    if col.startswith("setting_")
    or (
        col.startswith("sensor_")
        and "_rollmean_" not in col
    )
]
features.append("cycle")

X = df[features]
y = df["RUL"]

model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=15,
    max_depth=5,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1,
    n_jobs=-1,
)

model.fit(X, y)

model_path = MODEL_DIR / "sentinelx_lightgbm.joblib"
joblib.dump(
    {
        "model": model,
        "features": features,
    },
    model_path,
)

print("Training engines:", df["unit"].nunique())
print("Features:", len(features))
print("Model saved to:", model_path)
