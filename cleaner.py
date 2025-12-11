from PIL import Image
from pathlib import Path

ROOT = Path("data")

bad = 0

for path in ROOT.rglob("*.jpg"):
    try:
        img = Image.open(path)
        img.verify()
    except Exception:
        bad += 1
        path.unlink()

print("Битых файлов:", bad)
