import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from prepare_data import load_training_data


df = load_training_data("FD001")

# Consistent split by engine
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42,
)

train_idx, val_idx = next(
    splitter.split(df, groups=df["unit"])
)

train_df = df.iloc[train_idx]
val_df = df.iloc[val_idx]

# Original features only
features = [
    col for col in df.columns
    if col.startswith("setting_")
    or (
        col.startswith("sensor_")
        and "_rollmean_" not in col
    )
]
features.append("cycle")

X_train = train_df[features]
y_train = train_df["RUL"]
X_val = val_df[features]
y_val = val_df["RUL"]

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    ),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    ),
}

results = []

print("Training engines:", train_df["unit"].nunique())
print("Validation engines:", val_df["unit"].nunique())
print(
    "Shared engine IDs:",
    len(set(train_df["unit"]) & set(val_df["unit"])),
)

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    rmse = np.sqrt(
        mean_squared_error(y_val, predictions)
    )
    mae = mean_absolute_error(y_val, predictions)

    results.append({
        "Model": name,
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
    })

print("\n--- Final comparison ---")
print(
    pd.DataFrame(results)
    .sort_values("RMSE")
    .to_string(index=False)
)