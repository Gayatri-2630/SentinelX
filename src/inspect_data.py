from pathlib import Path

DATA_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "6. Turbofan Engine Degradation Simulation Data Set"
)

print("Dataset directory:", DATA_DIR)
print("Directory exists:", DATA_DIR.exists())

if DATA_DIR.exists():
    for file in sorted(DATA_DIR.iterdir()):
        if file.is_file():
            print(f"{file.name}: {file.stat().st_size:,} bytes")