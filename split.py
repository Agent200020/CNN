import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

SOURCE = Path("PetImages")  # ваша папка
DEST = Path("data")

def load_images(dir_path):
    return sorted([f for f in os.listdir(dir_path) 
                   if f.lower().endswith((".jpg",".png",".jpeg"))])

def split_and_copy(files, src_dir, class_name):
    # 70% train, 15% val, 15% test
    train_files, tmp = train_test_split(files, test_size=0.30, random_state=42)
    val_files, test_files = train_test_split(tmp, test_size=0.50, random_state=42)

    for subset, filelist in zip(["train","val","test"], 
                                [train_files, val_files, test_files]):
        outdir = DEST / subset / class_name
        outdir.mkdir(parents=True, exist_ok=True)
        for f in filelist:
            shutil.copy(src_dir / f, outdir / f)

# --- обработка кошек ---
cat_dir = SOURCE / "Cat"
cats = load_images(cat_dir)
split_and_copy(cats, cat_dir, "cats")

# --- обработка собак ---
dog_dir = SOURCE / "Dog"
dogs = load_images(dog_dir)
split_and_copy(dogs, dog_dir, "dogs")

print("Готово! Data folder создан.")
