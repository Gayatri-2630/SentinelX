import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor

from prepare_data import load_training_data


# Load FD001 training data
df = load_training_data("FD001")

# Use the same original features as the model comparison
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
groups = df["unit"]

# Five repeatable splits, with complete engines kept together
splitter = GroupShuffleSplit(
    n_splits=5,
    test_size=0.2,
    random_state=42,
)

results = []

for split_number, (train_idx, val_idx) in enumerate(
    splitter.split(X, y, groups=groups),
    start=1,
):
    train_engines = set(groups.iloc[train_idx])
    val_engines = set(groups.iloc[val_idx])
    shared_engines = train_engines & val_engines

    if shared_engines:
        raise ValueError("Engine leakage detected!")

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

    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    predictions = model.predict(X.iloc[val_idx])

    rmse = np.sqrt(
        mean_squared_error(y.iloc[val_idx], predictions)
    )
    mae = mean_absolute_error(y.iloc[val_idx], predictions)

    results.append({
        "Split": split_number,
        "Training engines": len(train_engines),
        "Validation engines": len(val_engines),
        "Shared engines": len(shared_engines),
        "RMSE": rmse,
        "MAE": mae,
    })

results_df = pd.DataFrame(results)

print("\n--- LightGBM Stability Evaluation: FD001 ---")
print(results_df.round(2).to_string(index=False))

print("\n--- Summary across five splits ---")
for metric in ["RMSE", "MAE"]:
    print(
        f"{metric}: "
        f"{results_df[metric].mean():.2f} "
        f"+/- {results_df[metric].std(ddof=1):.2f} cycles"
    )

print("\nNote: These are validation results, not official test results.")
