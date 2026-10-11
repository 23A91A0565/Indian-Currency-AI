# Model Checkpoints

This directory contains the trained PyTorch ResNet-34 model weights for Indian currency recognition.

## Active Model
- **`model_currency.pth`** (~85 MB): PyTorch ResNet-34 weights fine-tuned on Mahatma Gandhi (New) Series legal tender: ₹10, ₹20, ₹50, ₹100, ₹200, ₹500.

## Training / Regeneration
To train or retrain the model:
```bash
python train.py --epochs 5 --batch-size 16 --lr 0.0001 --output models/model_currency.pth
```
