
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

# Reproduce the exact 60/20/20 engine split.
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

# Compare each engine's final 30 RUL cycles with its earlier readings.
validation_df["phase"] = "Earlier readings"
validation_df.loc[
    validation_df["RUL"] <= 30, "phase"
] = "Final 30 RUL cycles"

engine_phase = validation_df.groupby(
    ["unit", "phase"], observed=False
)["score"].mean().unstack()

if "Earlier readings" not in engine_phase:
    raise ValueError("No earlier-reading group was available.")
if "Final 30 RUL cycles" not in engine_phase:
    raise ValueError("No final-30-cycle group was available.")

comparison = engine_phase.dropna(
    subset=["Earlier readings", "Final 30 RUL cycles"]
).copy()

comparison["score_change"] = (
    comparison["Final 30 RUL cycles"]
    - comparison["Earlier readings"]
)

print("Engines with both phases:", len(comparison))
print("\nPer-engine score comparison:")
print(comparison.round(5).to_string())

print("\nSummary across engines:")
print(
    comparison[
        ["Earlier readings", "Final 30 RUL cycles", "score_change"]
    ].agg(["mean", "median", "std", "min", "max"]).round(5).to_string()
)

print(
    "\nEngines with lower mean scores in their final 30 RUL cycles:",
    int((comparison["score_change"] < 0).sum()),
    "of",
    len(comparison),
)

print(
    "\nInterpretation: this measures score changes associated with "
    "late-life operation, not verified fault-detection accuracy."
)
