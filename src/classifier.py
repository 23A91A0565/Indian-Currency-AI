"""
classifier.py — Indian Banknote Classification Engine
=====================================================
Uses a deep learning convolutional neural network (ResNet-34) fine-tuned on
Mahatma Gandhi (New) Series Indian currency notes: ₹10, ₹20, ₹50, ₹100, ₹200, ₹500.

Beginner-Friendly Explanation:
1. ResNet-34 is a 34-layer residual network pre-trained on ImageNet.
2. It learns visual representations (color, pattern, motifs, texture).
3. The final classification head outputs probability scores for each note class.
"""

from pathlib import Path
from typing import Union, Dict, Any, Optional

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torchvision import models, transforms

from src.denominations import DENOMINATION_INFO, SUPPORTED_CLASSES


class CurrencyClassifier:
    """
    Inference engine for classifying Indian currency notes.
    Supports input as a file path, PIL Image, or NumPy array.
    """

    def __init__(
        self,
        checkpoint_path: str = "models/model_currency.pth",
        device: Optional[str] = None
    ):
        # Select GPU if available, otherwise CPU
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.checkpoint_path = Path(checkpoint_path)
        self.classes = list(SUPPORTED_CLASSES)

        # Standard PyTorch image preprocessing for ResNet models
        # - Resize image to 224x224 pixels
        # - Convert pixel values from [0, 255] to tensor [0.0, 1.0]
        # - Normalize using ImageNet dataset mean and standard deviation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Load model and weights
        self.model = self._load_model()
        self.model.eval()

    def _load_model(self) -> nn.Module:
        """
        Loads the ResNet-34 architecture and restores trained weights from disk.
        """
        target_checkpoint = self.checkpoint_path
        if not target_checkpoint.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at '{target_checkpoint}'. "
                "Please run `python train.py` to train the model."
            )

        # Load saved weights dictionary
        checkpoint_data = torch.load(target_checkpoint, map_location=self.device)

        # Extract classes and state dictionary if metadata was stored
        if isinstance(checkpoint_data, dict) and "classes" in checkpoint_data:
            self.classes = checkpoint_data["classes"]
            state_dict = checkpoint_data["model_state_dict"]
        elif isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
            state_dict = checkpoint_data["model_state_dict"]
        else:
            state_dict = checkpoint_data

        # Initialize ResNet-34 architecture without default pretrained weights
        model = models.resnet34(weights=None)
        in_features = model.fc.in_features

        # Replace classification layer to match the number of currency classes
        model.fc = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(in_features, len(self.classes))
        )

        # Load weights into model
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            # Fallback if checkpoint used a single Linear layer instead of Sequential
            model.fc = nn.Linear(in_features, len(self.classes))
            model.load_state_dict(state_dict)

        model = model.to(self.device)
        print(f"[*] CurrencyClassifier initialized on [{self.device}] for classes: {self.classes}")
        return model

    def _prepare_image(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
        """
        Normalizes various input formats (path, numpy array, PIL) into an RGB PIL Image.
        """
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input)
        elif isinstance(image_input, np.ndarray):
            # Check OpenCV BGR vs grayscale vs RGB array
            if len(image_input.shape) == 2:
                img = Image.fromarray(image_input).convert("RGB")
            elif image_input.shape[2] == 3:
                # Convert BGR (OpenCV format) to RGB
                rgb_array = image_input[:, :, ::-1]
                img = Image.fromarray(rgb_array)
            else:
                img = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # Automatically adjust orientation from mobile EXIF tags
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")

    def predict(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """
        Runs neural network inference on an input banknote image.

        Returns:
            Dict containing:
                - denomination: string (e.g. "₹500")
                - confidence: float percentage (e.g. 98.45)
                - raw_class: string label (e.g. "500")
                - name: string (e.g. "Five Hundred Rupees")
                - details: official RBI denomination details
                - probabilities: dict mapping class name to confidence percentage
        """
        pil_img = self._prepare_image(image_input)
        image_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # Run forward pass without calculating gradients (faster inference)
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

        # Find the highest probability class
        predicted_index = int(np.argmax(probabilities))
        predicted_class = str(self.classes[predicted_index])
        confidence_pct = float(probabilities[predicted_index] * 100.0)

        # Retrieve official banknote specifications
        details = DENOMINATION_INFO.get(predicted_class, {
            "denomination": f"₹{predicted_class}",
            "name": f"{predicted_class} Rupees",
            "color": "Unknown",
            "motif": "Unknown",
            "dimensions": "Unknown",
            "bleed_lines": "Unknown",
        })

        # Probability distribution across all supported denominations
        all_probabilities = {
            self.classes[i]: round(float(probabilities[i] * 100.0), 2)
            for i in range(len(self.classes))
        }

        return {
            "denomination": details.get("denomination", f"₹{predicted_class}"),
            "confidence": round(confidence_pct, 2),
            "raw_class": predicted_class,
            "name": details.get("name", f"{predicted_class} Rupees"),
            "details": details,
            "probabilities": all_probabilities,
        }
