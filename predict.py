"""
predict.py — Command-Line Banknote Inference Tool
=================================================
Runs deep learning classification on an input Indian banknote image.

Usage:
    python predict.py --image path/to/banknote.jpg
    python predict.py --image path/to/banknote.jpg --model models/model_currency.pth
"""

import sys
import argparse
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.classifier import CurrencyClassifier


def main():
    parser = argparse.ArgumentParser(
        description="RupeeLens: Indian Currency Note Classifier (ResNet-34)"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the banknote image to classify"
    )
    parser.add_argument(
        "--model",
        default="models/model_currency.pth",
        help="Path to trained PyTorch model weights (default: models/model_currency.pth)"
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[!] Error: Image not found at '{image_path}'")
        sys.exit(1)

    # Initialize classifier
    classifier = CurrencyClassifier(checkpoint_path=args.model)

    # Run prediction
    result = classifier.predict(str(image_path))

    # Display formatted results
    print("\n" + "=" * 55)
    print(" 🇮🇳 INDIAN CURRENCY RECOGNITION RESULT")
    print("=" * 55)
    print(f"  Denomination : {result['denomination']} ({result['name']})")
    print(f"  Confidence   : {result['confidence']:.2f}%")
    print(f"  Base Color   : {result['details'].get('color', 'N/A')}")
    print(f"  Motif        : {result['details'].get('motif', 'N/A')}")
    print(f"  Dimensions   : {result['details'].get('dimensions', 'N/A')}")
    print(f"  Bleed Lines  : {result['details'].get('bleed_lines', 'N/A')}")
    print("-" * 55)
    print("  Class Probabilities:")
    for cls, prob in sorted(result["probabilities"].items(), key=lambda x: int(x[0])):
        bar = "█" * int(prob / 4)
        print(f"    ₹{cls:<4} : {prob:5.1f}% | {bar}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
