import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from prepare_data import load_training_data


# Load the dataset
df = load_training_data("FD001")

# Split by complete engines
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42,
)

train_idx, val_idx = next(
    splitter.split(df, groups=df["unit"])
)

train_df = df.iloc[train_idx].copy()
val_df = df.iloc[val_idx].copy()

# Define two feature sets
sensor_features = [
    col for col in df.columns
    if col.startswith("setting_")
    or col.startswith("sensor_")
]

feature_sets = {
    "With cycle": sensor_features + ["cycle"],
    "Without cycle": sensor_features,
}


def evaluate_model(name, features):
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        train_df[features],
        train_df["RUL"],
    )

    predictions = model.predict(val_df[features])

    rmse = np.sqrt(
        mean_squared_error(val_df["RUL"], predictions)
    )
    mae = mean_absolute_error(
        val_df["RUL"], predictions
    )

    print(f"\n{name}")
    print(f"RMSE: {rmse:.2f} cycles")
    print(f"MAE:  {mae:.2f} cycles")

    return rmse, mae


print("Validation engines:", val_df["unit"].nunique())
print(
    "Shared engine IDs:",
    len(set(train_df["unit"]) & set(val_df["unit"])),
)

results = {}

for name, features in feature_sets.items():
    results[name] = evaluate_model(name, features)

print("\n--- Experiment comparison ---")
print(f"RMSE difference (without - with cycle): "
      f"{results['Without cycle'][0] - results['With cycle'][0]:.2f}")