import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor

from prepare_data import load_training_data


df = load_training_data("FD001")

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

experiments = [
    {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 5,
        "min_child_samples": 20,
    },
    {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": 5,
        "min_child_samples": 20,
    },
    {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
    },
    {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 5,
        "min_child_samples": 40,
    },
]

results = []

print("Training engines:", train_df["unit"].nunique())
print("Validation engines:", val_df["unit"].nunique())
print("Shared engine IDs:",
      len(set(train_df["unit"]) & set(val_df["unit"])))

for i, params in enumerate(experiments, start=1):
    model = LGBMRegressor(
        **params,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    mae = mean_absolute_error(y_val, predictions)

    results.append({
        "Experiment": i,
        **params,
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
    })

print("\n--- LightGBM tuning results ---")
results_df = pd.DataFrame(results).sort_values("RMSE")
print(results_df.to_string(index=False))

print("\nBest validation configuration:")
print(results_df.iloc[0].to_string())