import json
from pathlib import Path

from src.prepare_data import load_training_data

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "src" / "sensor_ranges.json"

FEATURE_COLUMNS = (
    ["setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

df = load_training_data("FD001")

ranges = {
    column: {
        "min": float(df[column].min()),
        "max": float(df[column].max()),
    }
    for column in FEATURE_COLUMNS
}

OUTPUT_PATH.write_text(json.dumps(ranges, indent=2), encoding="utf-8")

print(f"Exported {len(ranges)} feature ranges to {OUTPUT_PATH}")