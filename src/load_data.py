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

train_path = DATA_DIR / "train_FD001.txt"

train_df = pd.read_csv(
    train_path,
    sep=r"\s+",
    header=None,
    names=COLUMNS,
)

print("Dataset shape:", train_df.shape)
print("\nFirst five rows:")
print(train_df.head())

print("\nUnique engines:", train_df["unit"].nunique())
print("Total recorded cycles:", len(train_df))
print("Number of columns:", len(train_df.columns))

print("\nMissing values:", train_df.isnull().sum().sum())
print("Duplicate rows:", train_df.duplicated().sum())