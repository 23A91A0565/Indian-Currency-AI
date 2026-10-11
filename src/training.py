"""
training.py — ResNet-34 Transfer Learning Training Pipeline
===========================================================
Trains a PyTorch ResNet-34 classifier to recognize Mahatma Gandhi (New) Series
denominations: ₹10, ₹20, ₹50, ₹100, ₹200, ₹500.

Beginner-Friendly Concept:
- Transfer Learning: We take a deep neural network (ResNet-34) already trained
  on millions of images (ImageNet), freeze early feature extractors, and
  fine-tune the final classification layer to recognize Indian currency notes.
- Loss Function: CrossEntropyLoss measures error between predicted class & real label.
- Optimizer: Adam automatically adjusts learning rates for steady convergence.
"""

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split


def get_data_transforms():
    """
    Returns image transformations for training.
    Includes data augmentation to prevent overfitting on small datasets:
    - RandomRotation (+-15 degrees)
    - ColorJitter (brightness and contrast variation)
    - Normalization with ImageNet statistics
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def train_model(
    data_dir: str = "data",
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    output_path: str = "models/model_currency.pth",
    device: Optional[torch.device] = None
):
    """
    Executes the training and validation loops, saving the best performing model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Dataset folder '{data_dir}' not found. "
            "Please ensure images are organized into class folders under data/ (e.g., data/10, data/500)."
        )

    # Load dataset using torchvision ImageFolder
    transform = get_data_transforms()
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    classes = dataset.classes

    print("\n" + "=" * 50)
    print(" 🚀 STARTING MODEL TRAINING")
    print("=" * 50)
    print(f"[*] Detected Classes : {classes}")
    print(f"[*] Total Images     : {len(dataset)}")
    print(f"[*] Compute Device   : {device}")
    print(f"[*] Total Epochs     : {epochs}")
    print(f"[*] Batch Size       : {batch_size}")
    print(f"[*] Learning Rate    : {learning_rate}")
    print("-" * 50)

    # 80% train / 20% validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Build model using transfer learning on ResNet-34
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    in_features = model.fc.in_features
    # Replace final layer with the exact number of currency classes
    model.fc = nn.Linear(in_features, len(classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0
    Path("models").mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += torch.sum(predicted == labels).item()
            train_total += labels.size(0)

        train_acc = (train_correct / train_total) * 100 if train_total > 0 else 0.0
        avg_train_loss = running_loss / train_total if train_total > 0 else 0.0

        # 2. Validation Phase
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_correct += torch.sum(predicted == labels).item()
                val_total += labels.size(0)

        val_acc = (val_correct / val_total) * 100 if val_total > 0 else 0.0

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"-> Loss: {avg_train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}%"
        )

        # Save checkpoint if current validation accuracy is best so far
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "val_acc": val_acc,
                "epoch": epoch
            }, output_path)

    print("\n" + "=" * 50)
    print(" [OK] Training completed successfully!")
    print(f" [OK] Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f" [OK] Model weights saved to: {output_path}")
    print("=" * 50 + "\n")
    return best_val_acc
