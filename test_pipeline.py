"""
test_pipeline.py — End-to-End Pipeline Verification Test
========================================================
Runs automated checks to verify that:
1. PyTorch compute engine and torchvision ResNet-34 initialize properly.
2. Training mini-loop executes and saves model weights.
3. CurrencyClassifier loads weights and produces valid predictions.

Usage:
    python test_pipeline.py
"""

import os
import sys
import shutil
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.classifier import CurrencyClassifier
from src.denominations import SUPPORTED_CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = SUPPORTED_CLASSES
DUMMY_DIR = "dummy_test_data"


def create_synthetic_dataset():
    """Generates a miniature synthetic dataset for testing."""
    if os.path.exists(DUMMY_DIR):
        shutil.rmtree(DUMMY_DIR)

    for split in ["train", "val"]:
        for cls in CLASSES:
            folder = os.path.join(DUMMY_DIR, split, cls)
            os.makedirs(folder, exist_ok=True)
            for i in range(2):
                arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                Image.fromarray(arr).save(os.path.join(folder, f"sample_{i}.jpg"))


def run_test():
    print("\n" + "=" * 50)
    print(" [*] RUNNING PIPELINE VERIFICATION TEST")
    print("=" * 50)

    # 1. Test Dataset Creation
    create_synthetic_dataset()
    print("[*] Created synthetic test dataset.")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(os.path.join(DUMMY_DIR, "train"), transform=transform)
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

    print(f"[*] Detected Classes: {train_data.classes}")
    print(f"[*] Compute Device  : {DEVICE}")

    # 2. Test ResNet-34 Training Pass
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

    print("[*] Running 1 mini-epoch training pass...")
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

    os.makedirs("models", exist_ok=True)
    test_weights_path = "models/test_weights.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": CLASSES
    }, test_weights_path)
    print(f"[*] Saved test weights to {test_weights_path}")

    # 3. Test Inference with CurrencyClassifier
    print("[*] Verifying CurrencyClassifier inference...")
    classifier = CurrencyClassifier(checkpoint_path=test_weights_path)
    sample_img = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
    pred = classifier.predict(sample_img)

    assert "denomination" in pred, "Missing 'denomination' in prediction"
    assert "confidence" in pred, "Missing 'confidence' in prediction"
    assert "probabilities" in pred, "Missing 'probabilities' in prediction"
    print(f"[*] Inference Output: {pred['denomination']} ({pred['confidence']}%)")

    # Clean up dummy dataset and test weights
    if os.path.exists(DUMMY_DIR):
        shutil.rmtree(DUMMY_DIR)
    if os.path.exists(test_weights_path):
        os.remove(test_weights_path)

    print("\n" + "=" * 50)
    print(" [OK] Pipeline check passed! All components functional.")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_test()