
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

# Reproduce the existing 60/20/20 engine-level split.
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

candidate_quantiles = [0.01, 0.025, 0.05, 0.10]

for q in candidate_quantiles:
    threshold = validation_df["score"].quantile(q)
    evaluation = validation_df.copy()
    evaluation["flagged"] = evaluation["score"] <= threshold

    # Summarize alert behavior independently for each engine.
    summary = evaluation.groupby("unit").agg(
        readings=("flagged", "size"),
        flagged=("flagged", "sum"),
        last_cycle=("cycle", "max"),
    )

    summary["flagged_percent"] = (
        summary["flagged"] / summary["readings"] * 100
    ).round(2)

    earlier_flags = evaluation[
        (evaluation["RUL"] > 30) & evaluation["flagged"]
    ].groupby("unit").size()

    summary["flags_RUL_above_30"] = (
        earlier_flags.reindex(summary.index, fill_value=0)
    )

    first_flag = (
        evaluation[evaluation["flagged"]]
        .groupby("unit")["cycle"]
        .min()
    )

    summary["first_flag_cycle"] = first_flag.reindex(summary.index)

    print("\n" + "=" * 65)
    print(f"Candidate: lowest {q * 100:.1f}% of validation scores")
    print(f"Threshold: {threshold:.6f}")
    print(
        summary.sort_values(
            "flagged_percent", ascending=False
        ).to_string()
    )

    print(
        "\nEngines with any flags:",
        int((summary["flagged"] > 0).sum()),
        "of",
        len(summary),
    )
    print(
        "Engines with flags at RUL > 30:",
        int((summary["flags_RUL_above_30"] > 0).sum()),
        "of",
        len(summary),
    )

print(
    "\nInterpretation: cycle and RUL associations describe alert timing; "
    "they do not establish fault-detection accuracy."
)
