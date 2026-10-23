
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from prepare_data import load_training_data


ROOT = Path(__file__).resolve().parents[1]

df = load_training_data("FD001")

features = (
    [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Split by engine, not by individual reading.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(df, groups=df["unit"])
)

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

assert set(train_df["unit"]).isdisjoint(set(test_df["unit"]))

# Baseline A: fit on readings across all training lifecycles.
all_lifecycle_model = Pipeline([
    ("scaler", StandardScaler()),
    ("detector", IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )),
])

all_lifecycle_model.fit(train_df[features])

# Baseline B: fit only on early-life readings (RUL > 100).
# This is a proxy for normal operation, NOT verified healthy data.
early_df = train_df[train_df["RUL"] > 100].copy()

if early_df["unit"].nunique() < 10:
    raise ValueError(
        "Too few training engines contribute early-life readings."
    )

early_life_model = Pipeline([
    ("scaler", StandardScaler()),
    ("detector", IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )),
])

early_life_model.fit(early_df[features])

# Compare flag rates across RUL groups in the held-out engines.
results = []

for model_name, model in [
    ("All-lifecycle baseline", all_lifecycle_model),
    ("Early-life proxy baseline", early_life_model),
]:
    flags = model.predict(test_df[features])

    evaluation = test_df[["unit", "cycle", "RUL"]].copy()
    evaluation["flagged"] = flags == -1
    evaluation["model"] = model_name

    evaluation["rul_stage"] = pd.cut(
        evaluation["RUL"],
        bins=[-1, 30, 60, 100, float("inf")],
        labels=["0-30", "31-60", "61-100", "101+"],
    )

    summary = evaluation.groupby(
        "rul_stage", observed=False
    ).agg(
        readings=("flagged", "size"),
        flagged=("flagged", "sum"),
    )

    summary["flagged_percent"] = (
        summary["flagged"] / summary["readings"] * 100
    ).round(2)

    summary.insert(0, "model", model_name)
    results.append(summary.reset_index())

print("Training engines:", train_df["unit"].nunique())
print("Held-out engines:", test_df["unit"].nunique())
print("Early-life training engines:", early_df["unit"].nunique())
print("Early-life training readings:", len(early_df))

print("\nFlag rates by held-out RUL group:")
print(
    pd.concat(results, ignore_index=True).to_string(index=False)
)

print(
    "\nInterpretation: these are exploratory flag rates, "
    "not fault-detection accuracy."
)
print(
    "The early-life subset is only a proxy for normal operation; "
    "its readings are not verified fault-free."
)
