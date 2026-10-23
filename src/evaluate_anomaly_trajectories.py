
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from prepare_data import load_training_data


df = load_training_data("FD001")

features = (
    [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Reproduce the same 60/20/20 split used in the previous experiment.
first_split = GroupShuffleSplit(
    n_splits=1, test_size=0.4, random_state=42
)
train_idx, remainder_idx = next(
    first_split.split(df, groups=df["unit"])
)

train_df = df.iloc[train_idx].copy()
remainder_df = df.iloc[remainder_idx].copy()

second_split = GroupShuffleSplit(
    n_splits=1, test_size=0.5, random_state=43
)
validation_idx, _ = next(
    second_split.split(remainder_df, groups=remainder_df["unit"])
)

validation_df = remainder_df.iloc[validation_idx].copy()

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    (
        "detector",
        IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        ),
    ),
])

pipeline.fit(train_df[features])

validation_df["score"] = pipeline.decision_function(
    validation_df[features]
)

# RUL is used only to describe lifecycle stages, not as an anomaly label.
validation_df["rul_stage"] = pd.cut(
    validation_df["RUL"],
    bins=[-1, 30, 60, 100, 150, float("inf")],
    labels=["0-30", "31-60", "61-100", "101-150", "151+"],
)

print("Validation engines:", validation_df["unit"].nunique())
print("\nScore distribution by RUL stage:")

summary = validation_df.groupby(
    "rul_stage", observed=False
)["score"].agg(["count", "mean", "median", "std", "min", "max"])

print(summary.round(5).to_string())

# Paired engine-level comparison: final 30 RUL cycles vs earlier readings.
validation_df["phase"] = (
    validation_df["RUL"] <= 30
).map({
    True: "Final 30 RUL cycles",
    False: "Earlier readings",
})

engine_means = validation_df.groupby(
    ["unit", "phase"], observed=False
)["score"].mean().unstack()

engine_means = engine_means.dropna(
    subset=["Earlier readings", "Final 30 RUL cycles"]
)

engine_means["score_change"] = (
    engine_means["Final 30 RUL cycles"]
    - engine_means["Earlier readings"]
)

print("\nPaired score change by engine:")
print(engine_means.round(5).to_string())

print("\nEngine-level score-change summary:")
print(engine_means["score_change"].describe().round(5).to_string())

print(
    "\nEngines with lower final-30-cycle mean scores:",
    int((engine_means["score_change"] < 0).sum()),
    "of",
    len(engine_means),
)

print(
    "\nThese are descriptive lifecycle associations, "
    "not verified anomaly-detection accuracy or lead-time estimates."
)
