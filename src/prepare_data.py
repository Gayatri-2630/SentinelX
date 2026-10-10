from pathlib import Path
import pandas as pd

DATA_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "6. Turbofan Engine Degradation Simulation Data Set"
)

COLUMNS = (
    ["unit", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_training_data(subset="FD001"):
    path = DATA_DIR / f"train_{subset}.txt"

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )

    max_cycles = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = max_cycles - df["cycle"]

    return df


if __name__ == "__main__":
    df = load_training_data("FD001")

    print("Shape:", df.shape)
    print("\nSample RUL labels:")
    print(df[["unit", "cycle", "RUL"]].head(10).to_string(index=False))

    print("\nRUL statistics:")
    print(df["RUL"].describe())

    print("\nTarget checks:")
    print("Missing RUL labels:", df["RUL"].isna().sum())
    print("Minimum RUL:", df["RUL"].min())
    print("Maximum RUL:", df["RUL"].max())
    print(
        "Rows where RUL = 0:",
        (df["RUL"] == 0).sum(),
    )