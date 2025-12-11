import torch
from pathlib import Path
import pandas as pd

RESULTS_ROOT = Path("results")

# Список для хранения результатов всех экспериментов
results = []

for exp_dir in RESULTS_ROOT.iterdir():
    model_file = exp_dir / "models" / "best_model.pth"
    if not model_file.exists():
        continue

    checkpoint = torch.load(model_file, map_location="cpu")
    val_acc = checkpoint.get("val_acc", None)
    if val_acc is None:
        continue

    results.append({
        "experiment": exp_dir.name,
        "val_acc": val_acc,
        "model_path": str(model_file)
    })

# Превращаем в DataFrame
df = pd.DataFrame(results)

# Сортируем по точности
df = df.sort_values(by="val_acc", ascending=False).reset_index(drop=True)

# Добавляем сводную статистику
stats = {
    "mean_val_acc": df["val_acc"].mean(),
    "median_val_acc": df["val_acc"].median(),
    "max_val_acc": df["val_acc"].max(),
    "min_val_acc": df["val_acc"].min()
}

print("\n==============================")
print(" ВСЕ ЭКСПЕРИМЕНТЫ ")
print("==============================")
print(df.to_string(index=False))

print("\n==============================")
print(" СВОДНАЯ СТАТИСТИКА ")
print("==============================")
for k, v in stats.items():
    print(f"{k}: {v:.4f}")

# ЛУЧШИЙ эксперимент
best_exp_row = df.iloc[0]
print("\n==============================")
print(" ЛУЧШИЙ ЭКСПЕРИМЕНТ ")
print("==============================")
print("Папка эксперимента:", best_exp_row["experiment"])
print("Путь к лучшей модели:", best_exp_row["model_path"])
print(f"Val accuracy: {best_exp_row['val_acc']:.4f}")
print("==============================\n")
