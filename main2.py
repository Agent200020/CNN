# Основные библиотеки
import os
import time
import json
from pathlib import Path
from datetime import datetime

# Вспомогательные библиотеки
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from PIL import Image
import matplotlib.pyplot as plt

# Конфиг для настройки параметров обучения

CFG = {

    "data_dir": "data",
    "img_size": 128,
    "batch_size": 64,
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,

    "conv_filters": [32, 64, 128],

    # augmentation toggle
    "use_augmentation": True,

    # training
    "epochs": 12,

    # optimizer config
    "optimizer": "SGD",   # "Adam" or "SGD"
    "lr": 1e-3,
    "momentum": 0.9,       # only for SGD

    # where to save results
    "results_root": "results",

    "activation": "ReLU"  # ReLU, LeakyReLU, GELU, ELU, Mish, Sigmoid, Tanh

}

# =============================
# Вспомогательные функции
# =============================
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_exp_name(cfg):
    arch = "conv{}".format(len(cfg["conv_filters"])) + "_" + "-".join(map(str, cfg["conv_filters"]))
    aug = "augOn" if cfg["use_augmentation"] else "augOff"
    opt = f'{cfg["optimizer"]}_lr{cfg["lr"]}'
    bs = f'bs{cfg["batch_size"]}'
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{arch}__{aug}__{opt}__{bs}__{ts}"
    return name.replace(" ", "")

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def make_activation(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "leakyrelu":
        return nn.LeakyReLU(0.01, inplace=True)
    elif name == "gelu":
        return nn.GELU()
    elif name == "elu":
        return nn.ELU(inplace=True)
    elif name == "mish":
        return nn.Mish()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unknown activation: {name}")


# =============================
# Конструктор модели
# =============================
class DynamicConvNet(nn.Module):
    def __init__(self, conv_filters, img_size, num_classes=2, activation="ReLU"):
        act = make_activation(activation)
        super().__init__()
        layers = []
        in_ch = 3
        for out_ch in conv_filters:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(make_activation(activation))
            layers.append(nn.MaxPool2d(kernel_size=2))  # halves spatial dims
            in_ch = out_ch
        layers.append(nn.Dropout(0.25))
        self.feature = nn.Sequential(*layers)
        # compute linear input size by forwarding a dummy tensor
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            feat = self.feature(dummy)
            self._flattened_size = int(np.prod(feat.shape[1:]))
        # classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flattened_size, 256),
            make_activation(activation),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        return x

# =============================
# Запуск эксперимента
# =============================
def run_experiment(cfg):
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    print(f"Using device: {device}")

    # Build experiment name and folders
    exp_name = make_exp_name(cfg)
    result_dir = Path(cfg["results_root"]) / exp_name
    models_dir = result_dir / "models"
    figures_dir = result_dir / "figures"
    ensure_dir(models_dir)
    ensure_dir(figures_dir)

    # Save config to file
    with open(result_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # Log header
    print("=== EXPERIMENT CONFIGURATION ===")
    for k, v in cfg.items():
        print(f"{k}: {v}")
    print("Results will be saved in:", str(result_dir))
    print("==============================\n")

    # Transforms
    if cfg["use_augmentation"]:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(cfg["img_size"]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((cfg["img_size"], cfg["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])

    test_tf = transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])

    # Datasets & loaders
    data_dir = Path(cfg["data_dir"])
    train_ds = datasets.ImageFolder(str(data_dir / "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(data_dir / "val"), transform=test_tf)
    test_ds  = datasets.ImageFolder(str(data_dir / "test"), transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val_loader   = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    test_loader  = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    class_names = train_ds.classes
    num_classes = len(class_names)
    print("Classes:", class_names)

    # Build model
    model = DynamicConvNet(
        cfg["conv_filters"],
        cfg["img_size"],
        num_classes=num_classes,
        activation=cfg["activation"]
    ).to(device)

    print(model)
    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    if cfg["optimizer"].lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg.get("momentum", 0.9))
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    # Training bookkeeping
    best_val_acc = 0.0
    best_model_path = None
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # Training loop
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            total += imgs.size(0)

        train_loss = running_loss / len(train_ds)
        train_acc = running_correct / len(train_ds)

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_running_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                val_running_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)

        val_loss = val_running_loss / len(val_ds)
        val_acc = val_running_correct / len(val_ds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = models_dir / "best_model.pth"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "cfg": cfg,
                "epoch": epoch,
                "val_acc": val_acc
            }, str(best_model_path))

        t1 = time.time()
        print(f"Epoch {epoch}/{cfg['epochs']}  Train loss: {train_loss:.4f}, acc: {train_acc:.4f}  |  Val loss: {val_loss:.4f}, acc: {val_acc:.4f}  time: {t1-t0:.1f}s")

        # Save periodic checkpoints (optional)
        checkpoint_path = models_dir / f"checkpoint_epoch{epoch}.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc
        }, str(checkpoint_path))

    # Save history to JSON
    with open(result_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # -----------------------
    # Отрисовка графиков
    # -----------------------
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history["train_loss"], label='train_loss')
    plt.plot(history["val_loss"], label='val_loss')
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history["train_acc"], label='train_acc')
    plt.plot(history["val_acc"], label='val_acc')
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    curves_path = figures_dir / "training_curves.png"
    plt.savefig(str(curves_path))
    plt.close()

    # -----------------------
    # Сохранение перепутанных картинок из матрицы ошибок
    # -----------------------
    misclassified_dir = figures_dir / "misclassified"
    ensure_dir(misclassified_dir)

    mis_images = []   

    test_samples = test_ds.samples  

    idx = 0  
    for imgs, labels in test_loader:
        batch_size = imgs.size(0)
        outputs = model(imgs.to(device))
        preds = outputs.argmax(dim=1).cpu()

        for i in range(batch_size):
            if preds[i].item() != labels[i].item():
                filepath, true_class = test_samples[idx]
                mis_images.append((imgs[i], labels[i].item(), preds[i].item(), filepath))
            idx += 1


    # Загрузка лучшей модели перед тестом
    if best_model_path is not None and best_model_path.exists():
        checkpoint = torch.load(str(best_model_path), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true = []
    y_pred = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_pred.extend(list(preds))
            y_true.extend(list(labels.numpy()))
    # Матрица ошибок
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax, xticks_rotation='vertical', cmap="Blues")
    ax.set_title(f"Confusion matrix\n{exp_name}")
    cm_path = figures_dir / "confusion_matrix.png"
    fig.savefig(str(cm_path))
    plt.close(fig)

    def save_misclassified_examples(mis_images, out_dir, class_names, max_to_save=50):
        for i, (img_tensor, true_label, pred_label, filepath) in enumerate(mis_images[:max_to_save]):
            # Нормализация
            img_np = img_tensor.numpy()
            img_np = (img_np * 0.5) + 0.5
            img_np = np.transpose(img_np, (1, 2, 0))
            img_np = np.clip(img_np, 0, 1)

            plt.figure(figsize=(3,3))
            plt.imshow(img_np)
            plt.title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}")
            plt.axis("off")

            base = os.path.basename(filepath)
            save_path = out_dir / f"{i:03d}_{class_names[true_label]}_as_{class_names[pred_label]}_{base}"
            plt.savefig(str(save_path))
            plt.close()

    save_misclassified_examples(mis_images, misclassified_dir, class_names)

    # -----------------------
    # Сохранение ряда предсказаний
    # -----------------------
    def save_prediction_examples(dl, path, n_images=8):
        imgs_batch, labels_batch = next(iter(dl))
        outputs = model(imgs_batch.to(device))
        preds = outputs.argmax(dim=1).cpu().numpy()
        # unnormalize and plot
        imgs_np = imgs_batch.numpy()
        # imgs are normalized with mean=0.5, std=0.5 -> original = img*std + mean
        imgs_np = (imgs_np * 0.5) + 0.5  # shape (B,C,H,W)
        n = min(n_images, imgs_np.shape[0])
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        fig = plt.figure(figsize=(cols * 3, rows * 3))
        for i in range(n):
            ax = fig.add_subplot(rows, cols, i+1)
            img = np.transpose(imgs_np[i], (1,2,0))
            ax.imshow(np.clip(img, 0, 1))
            ax.set_title(f"GT: {class_names[labels_batch[i]]}\nPred: {class_names[preds[i]]}")
            ax.axis('off')
        fig.savefig(str(path))
        plt.close(fig)

    examples_path = figures_dir / "examples.png"
    save_prediction_examples(test_loader, examples_path, n_images=8)

    # Отладочный лог
    final_msg = f"Experiment finished.\nBest val accuracy: {best_val_acc:.4f}"
    if best_model_path:
        final_msg += f"\nBest model saved to: {best_model_path}"
    final_msg += f"\nAll results saved in: {result_dir}"
    print(final_msg)
    return {
        "exp_name": exp_name,
        "result_dir": str(result_dir),
        "best_val_acc": best_val_acc,
        "best_model_path": str(best_model_path) if best_model_path is not None else None
    }

