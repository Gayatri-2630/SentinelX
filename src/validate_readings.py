import json
from pathlib import Path

RANGES_PATH = Path(__file__).resolve().parent / "sensor_ranges.json"

with RANGES_PATH.open("r", encoding="utf-8") as file:
    TRAINING_RANGES = json.load(file)


def get_training_ranges():
    return TRAINING_RANGES


def inspect_reading(reading):
    warnings = []

    for column, limits in TRAINING_RANGES.items():
        value = float(reading[column])

        if value < limits["min"] or value > limits["max"]:
            warnings.append({
                "feature": column,
                "value": value,
                "training_min": limits["min"],
                "training_max": limits["max"],
                "message": "Outside observed FD001 training range",
            })

    return {
        "within_observed_ranges": len(warnings) == 0,
        "warning_count": len(warnings),
        "warnings": warnings,
    }


if __name__ == "__main__":
    print(f"Loaded {len(get_training_ranges())} feature ranges.")
    print("Validation metadata ready.")