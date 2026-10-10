
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor

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

feature_cols = [
    col for col in df.columns
    if col.startswith("setting_")
    or col.startswith("sensor_")
]
feature_cols.append("cycle")

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)

model.fit(train_df[feature_cols], train_df["RUL"])

importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

print("\nTop 10 important features:")
print(importance.head(10).to_string(index=False))

plt.figure(figsize=(10, 6))
top = importance.head(10).sort_values("importance")
plt.barh(top["feature"], top["importance"])
plt.xlabel("Feature importance")
plt.title("SentinelX — Top 10 Feature Importances")
plt.tight_layout()
plt.savefig("models/feature_importance.png", dpi=150)
plt.show()