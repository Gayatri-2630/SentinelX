import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from prepare_data import load_training_data
from feature_engineering import add_rolling_features


# 1. Load and engineer features
df = load_training_data("FD001")
df = add_rolling_features(df)

# Safety check: all DataFrame columns must be unique
if not df.columns.is_unique:
    raise ValueError("Duplicate DataFrame column names detected!")

# 2. Define BASE features.
# IMPORTANT: Exclude rolling features from this list.
base_features = [
    col for col in df.columns
    if (
        col.startswith("setting_")
        or (
            col.startswith("sensor_")
            and "_rollmean_" not in col
        )
    )
]
base_features.append("cycle")

# 3. Define rolling features separately
rolling_features = [
    col for col in df.columns
    if "_rollmean_" in col
]

# 4. Validate the feature lists
combined_features = base_features + rolling_features

if len(base_features) != len(set(base_features)):
    raise ValueError("Duplicate base features detected!")

if len(combined_features) != len(set(combined_features)):
    raise ValueError("Duplicate features in combined feature list!")

print("Total rows:", len(df))
print("Unique DataFrame columns:", df.columns.is_unique)
print("Base feature count:", len(base_features))
print("Rolling feature count:", len(rolling_features))
print("Combined feature count:", len(combined_features))

# Expected: 25 base features, 63 rolling features, 88 combined.
assert len(base_features) == 25, "Unexpected base feature count"
assert len(rolling_features) == 63, "Unexpected rolling feature count"
assert len(combined_features) == 88, "Unexpected combined feature count"

# 5. Split by complete engines
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

shared_engines = (
    set(train_df["unit"]) & set(val_df["unit"])
)

print("\nTraining engines:", train_df["unit"].nunique())
print("Validation engines:", val_df["unit"].nunique())
print("Shared engine IDs:", len(shared_engines))

assert len(shared_engines) == 0, "Engine leakage detected!"


# 6. Train and evaluate models consistently
def evaluate(name, features):
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    X_train = train_df[features]
    y_train = train_df["RUL"]
    X_val = val_df[features]
    y_val = val_df["RUL"]

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    rmse = np.sqrt(
        mean_squared_error(y_val, predictions)
    )
    mae = mean_absolute_error(y_val, predictions)

    print(f"\n{name}")
    print("Features:", len(features))
    print(f"RMSE: {rmse:.2f} cycles")
    print(f"MAE:  {mae:.2f} cycles")

    return {"RMSE": rmse, "MAE": mae}


base_results = evaluate(
    "Model A — Without rolling features",
    base_features,
)

enhanced_results = evaluate(
    "Model B — With rolling features",
    combined_features,
)

print("\n--- Final comparison ---")
print(
    pd.DataFrame(
        [
            {"Model": "Without rolling", **base_results},
            {"Model": "With rolling", **enhanced_results},
        ]
    ).round(2).to_string(index=False)
)

print(
    "\nRMSE change (with - without): "
    f"{enhanced_results['RMSE'] - base_results['RMSE']:+.2f} cycles"
)

print(
    "MAE change (with - without): "
    f"{enhanced_results['MAE'] - base_results['MAE']:+.2f} cycles"
)
