import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PREDICTIONS_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "fd001_test_predictions.csv"
)

df = pd.read_csv(PREDICTIONS_PATH)

# Positive error = overestimation of remaining life
df["signed_error"] = (
    df["predicted_RUL"] - df["actual_RUL"]
)

df["absolute_error"] = df["signed_error"].abs()

print("Number of test engines:", len(df))
print(f"Mean signed error: {df['signed_error'].mean():.2f} cycles")
print(f"Median absolute error: {df['absolute_error'].median():.2f} cycles")
print(f"90th percentile absolute error: "
      f"{df['absolute_error'].quantile(0.90):.2f} cycles")

print("\nLargest absolute errors:")
print(
    df.nlargest(10, "absolute_error")[
        ["unit", "actual_RUL", "predicted_RUL",
         "signed_error", "absolute_error"]
    ].round(2).to_string(index=False)
)

print("\nMean signed error by actual RUL range:")
df["RUL_range"] = pd.cut(
    df["actual_RUL"],
    bins=[-1, 50, 100, 150, float("inf")],
    labels=["0-50", "51-100", "101-150", "151+"],
)

print(
    df.groupby("RUL_range", observed=False)["signed_error"]
    .agg(["count", "mean", "median"])
    .round(2)
)

# Plot actual vs predicted RUL
output_dir = PREDICTIONS_PATH.parent

plt.figure(figsize=(8, 6))
plt.scatter(df["actual_RUL"], df["predicted_RUL"], alpha=0.75)

lower = min(df["actual_RUL"].min(), df["predicted_RUL"].min())
upper = max(df["actual_RUL"].max(), df["predicted_RUL"].max())

plt.plot([lower, upper], [lower, upper], linestyle="--")
plt.xlabel("Actual RUL (cycles)")
plt.ylabel("Predicted RUL (cycles)")
plt.title("SentinelX — Official FD001 Test Predictions")
plt.tight_layout()

plot_path = output_dir / "fd001_actual_vs_predicted.png"
plt.savefig(plot_path, dpi=150)
plt.close()

print(f"\nSaved plot: {plot_path}")
