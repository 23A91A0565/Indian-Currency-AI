"""
data_prep.py — Dataset Preparation & Organization Utility
=========================================================
Organizes downloaded raw currency image datasets (e.g., from Kaggle/YOLO format)
into clean class folders: data/10, data/20, data/50, data/100, data/200, data/500.

Beginner-Friendly Explanation:
PyTorch's ImageFolder expects files organized as:
    data/
      10/  (images of ₹10)
      20/  (images of ₹20)
      ...
This utility scans the raw dataset folders, parses labels, and copies images
into their corresponding denomination folders.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

# Standard class mapping for the YOLO-annotated currency dataset
YOLO_CLASS_NAMES = ["10", "100", "20", "200", "50", "500"]
TARGET_CLASSES = ["10", "20", "50", "100", "200", "500"]


def get_class_from_yolo_label(label_file_path: str) -> Optional[str]:
    """
    Reads the first integer from a YOLO .txt annotation file to identify the class.
    """
    if not os.path.exists(label_file_path) or os.path.getsize(label_file_path) == 0:
        return None

    with open(label_file_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        if not first_line:
            return None
        try:
            class_idx = int(first_line.split()[0])
            if 0 <= class_idx < len(YOLO_CLASS_NAMES):
                return YOLO_CLASS_NAMES[class_idx]
        except (ValueError, IndexError):
            return None

    return None


def organize_raw_dataset(
    raw_dir: str = os.path.join("raw_data", "Indian currency"),
    output_dir: str = "data"
) -> Dict[str, int]:
    """
    Scans the raw data directory and organizes images into target class folders.
    """
    if not os.path.exists(raw_dir):
        print(f"[!] Source folder '{raw_dir}' does not exist.")
        return {}

    # Create destination class directories
    for class_name in TARGET_CLASSES:
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

    counts: Dict[str, int] = {c: 0 for c in TARGET_CLASSES}
    total_images_processed = 0

    # Scan dataset subfolders (train, test, valid)
    for split_folder in os.listdir(raw_dir):
        split_path = os.path.join(raw_dir, split_folder)
        if not os.path.isdir(split_path):
            continue

        images_dir = os.path.join(split_path, "images")
        labels_dir = os.path.join(split_path, "labels")

        # Fallback if label directory is named singular 'label'
        if not os.path.exists(labels_dir) and os.path.exists(os.path.join(split_path, "label")):
            labels_dir = os.path.join(split_path, "label")

        if not os.path.exists(images_dir):
            continue

        print(f"[*] Processing split: {split_folder}...")

        for img_file in os.listdir(images_dir):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            base_name = os.path.splitext(img_file)[0]
            label_file = os.path.join(labels_dir, base_name + ".txt")

            detected_class = get_class_from_yolo_label(label_file)
            if detected_class and detected_class in TARGET_CLASSES:
                src_path = os.path.join(images_dir, img_file)
                unique_filename = f"{split_folder}_{img_file}"
                dst_path = os.path.join(output_dir, detected_class, unique_filename)

                shutil.copy2(src_path, dst_path)
                counts[detected_class] += 1
                total_images_processed += 1

    print("\n" + "=" * 45)
    print("      DATASET ORGANIZATION COMPLETE")
    print("=" * 45)
    for c in TARGET_CLASSES:
        print(f"  data/{c:<4} : {counts[c]:4d} images")
    print("-" * 45)
    print(f"  Total Images Sorted: {total_images_processed}")
    print("=" * 45 + "\n")

    return counts
