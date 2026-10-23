
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

# Reproduce the existing 60/20/20 engine split.
split_a = GroupShuffleSplit(
    n_splits=1, test_size=0.4, random_state=42
)
train_idx, remainder_idx = next(
    split_a.split(df, groups=df["unit"])
)

train_df = df.iloc[train_idx].copy()
remainder_df = df.iloc[remainder_idx].copy()

split_b = GroupShuffleSplit(
    n_splits=1, test_size=0.5, random_state=43
)
validation_idx, _ = next(
    split_b.split(remainder_df, groups=remainder_df["unit"])
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

validation_df["rul_stage"] = pd.cut(
    validation_df["RUL"],
    bins=[-1, 30, 60, 100, float("inf")],
    labels=["0-30", "31-60", "61-100", "101+"],
)

# Exploratory candidates only. Do not choose a threshold
# from the reserved final-test engines.
candidate_quantiles = [0.01, 0.025, 0.05, 0.10]

print("Validation engines:", validation_df["unit"].nunique())
print("Validation readings:", len(validation_df))

for q in candidate_quantiles:
    threshold = validation_df["score"].quantile(q)
    flagged = validation_df["score"] <= threshold

    evaluation = validation_df[["unit", "rul_stage"]].copy()
    evaluation["flagged"] = flagged.to_numpy()

    print(f"\n{'=' * 55}")
    print(f"Candidate: lowest {q * 100:.1f}% of validation scores")
    print(f"Threshold: {threshold:.6f}")
    print(f"Overall flagged readings: {flagged.mean() * 100:.2f}%")

    summary = evaluation.groupby(
        "rul_stage", observed=False
    ).agg(
        readings=("flagged", "size"),
        flagged=("flagged", "sum"),
    )
    summary["flagged_percent"] = (
        summary["flagged"] / summary["readings"] * 100
    ).round(2)

    print("\nFlag rate by RUL stage:")
    print(summary.to_string())

    # An engine is considered to have an earlier-life alert if
    # at least one reading with RUL > 30 is flagged.
    earlier = evaluation[
        evaluation["rul_stage"] != "0-30"
    ]

    engine_alerts = earlier.groupby("unit")["flagged"].any()

    print(
        "\nEngines with at least one flag at RUL > 30:",
        int(engine_alerts.sum()),
        "of",
        len(engine_alerts),
    )

print(
    "\nThese are exploratory flag rates, not fault-detection "
    "accuracy or verified false-alarm rates."
)
