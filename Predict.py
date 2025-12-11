# predict.py

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse
import json
from pathlib import Path
from main2 import DynamicConvNet   # <-- модель из твоего train-кода

# ----------------------------------------------------------
# Загрузка модели
# ----------------------------------------------------------
def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)

    cfg = checkpoint["cfg"]  # конфиг из обучения
    conv_filters = cfg["conv_filters"]
    img_size = cfg["img_size"]

    model = DynamicConvNet(conv_filters, img_size, num_classes=2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, cfg


# ----------------------------------------------------------
# Предобработка изображения
# ----------------------------------------------------------
def preprocess(image_path, img_size):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])
    img = Image.open(image_path).convert("RGB")
    return tf(img).unsqueeze(0)


# ----------------------------------------------------------
# Предсказание
# ----------------------------------------------------------
def predict(model, tensor, class_names):
    with torch.no_grad():
        logits = model(tensor)
        pred = logits.argmax(dim=1).item()
        return class_names[pred]


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
if __name__ == "__main__":
    model_path = "results/conv3_64-128-256__augOff__Adam_lr0.0005__bs32__20251123-135703/models/best_model.pth"
    image_path = "PetImages/examples/dog_pred.jpg"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, cfg = load_model(model_path, device)
    class_names = ["cat", "dog"]

    tensor = preprocess(image_path, cfg["img_size"]).to(device)
    label = predict(model, tensor, class_names)

    print("\n======================")
    print(" Prediction:", label.upper())
    print("======================\n")
