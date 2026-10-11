"""
train.py — Model Training Script Entry Point
============================================
Fine-tunes a PyTorch ResNet-34 classifier using transfer learning on the
Mahatma Gandhi (New) Series dataset in the data/ directory.

Usage:
    python train.py
    python train.py --epochs 10 --batch-size 16 --lr 0.0001
"""

import sys
import argparse

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.training import train_model


def main():
    parser = argparse.ArgumentParser(
        description="Train Indian Currency Note Classifier (ResNet-34)"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing class folders of images (default: data)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size (default: 16)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for Adam optimizer (default: 0.0001)"
    )
    parser.add_argument(
        "--output",
        default="models/model_currency.pth",
        help="Output path for trained model weights (default: models/model_currency.pth)"
    )
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_path=args.output
    )


if __name__ == "__main__":
    main()