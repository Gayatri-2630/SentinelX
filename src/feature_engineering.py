import pandas as pd

from prepare_data import load_training_data


def add_rolling_features(df):
    df = df.sort_values(
        ["unit", "cycle"]
    ).copy()

    sensor_cols = [
        col for col in df.columns
        if col.startswith("sensor_")
    ]

    for window in [3, 5, 10]:
        rolling = (
            df.groupby("unit", sort=False)[sensor_cols]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .sort_index()
        )

        rolling.columns = [
            f"{col}_rollmean_{window}"
            for col in sensor_cols
        ]

        df[rolling.columns] = rolling

    return df


if __name__ == "__main__":
    df = load_training_data("FD001")
    enhanced_df = add_rolling_features(df)

    print("Original shape:", df.shape)
    print("Enhanced shape:", enhanced_df.shape)
    print("Unique column names:", enhanced_df.columns.is_unique)

    example_cols = [
        "unit",
        "cycle",
        "sensor_11",
        "sensor_11_rollmean_3",
        "sensor_11_rollmean_5",
        "sensor_11_rollmean_10",
    ]

    print("\nExample rolling features:")
    print(enhanced_df[example_cols].head(10).to_string(index=False))

    print(
        "\nMissing rolling-feature values:",
        enhanced_df.filter(like="_rollmean_").isna().sum().sum(),
    )
