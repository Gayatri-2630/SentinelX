import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prepare_data import load_training_data

# 1. Load data
df = load_training_data("FD001")

# 2. Split by complete engines
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

# 3. Select features (never use engine ID as a feature)
feature_cols = [
    col for col in df.columns
    if col.startswith("setting_")
    or col.startswith("sensor_")
]
feature_cols.append("cycle")

X_train = train_df[feature_cols]
y_train = train_df["RUL"]

X_val = val_df[feature_cols]
y_val = val_df["RUL"]

# 4. Train a simple baseline
dummy = DummyRegressor(strategy="median")
dummy.fit(X_train, y_train)
dummy_preds = dummy.predict(X_val)

# 5. Train a Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)
preds = model.predict(X_val)

# 6. Evaluate both models
def evaluate(name, actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    print(f"\n{name}")
    print(f"RMSE: {rmse:.2f} cycles")
    print(f"MAE:  {mae:.2f} cycles")

print("Training engines:", train_df["unit"].nunique())
print("Validation engines:", val_df["unit"].nunique())
print("Shared engine IDs:",
      len(set(train_df["unit"]) & set(val_df["unit"])))

evaluate("Median baseline", y_val, dummy_preds)
evaluate("Random Forest", y_val, preds)