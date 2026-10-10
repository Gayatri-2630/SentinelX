from prepare_data import load_training_data

df = load_training_data("FD001")

sensor_cols = [f"sensor_{i}" for i in range(1, 22)]

print("\nSENSOR RANGES — FD001 TRAINING DATA\n")

print(
    df[sensor_cols]
    .describe()
    .loc[["min", "max", "mean", "std"]]
    .round(3)
    .to_string()
)

print("\nOPERATING SETTING RANGES\n")

setting_cols = ["setting_1", "setting_2", "setting_3"]

print(
    df[setting_cols]
    .describe()
    .loc[["min", "max", "mean", "std"]]
    .round(3)
    .to_string()
)
