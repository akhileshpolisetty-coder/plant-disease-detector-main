# generate_class_names.py
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dataset import get_dataloaders

_, _, classes = get_dataloaders()

Path("models").mkdir(exist_ok=True)
with open("models/class_names.json", "w") as f:
    json.dump(classes, f)

print("Saved classes:", classes)