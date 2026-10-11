"""
prepare_data.py — Dataset Preparation Script Entry Point
========================================================
Scans raw currency image folders and organizes them into clean class directories
(data/10, data/20, data/50, data/100, data/200, data/500).

Usage:
    python prepare_data.py
    python prepare_data.py --source raw_data/"Indian currency" --target data
"""

import os
import sys
import argparse

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.data_prep import organize_raw_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Organize raw Indian currency images into class folders"
    )
    parser.add_argument(
        "--source",
        default=os.path.join("raw_data", "Indian currency"),
        help="Path to raw currency dataset directory"
    )
    parser.add_argument(
        "--target",
        default="data",
        help="Target directory to place organized class folders (default: data)"
    )
    args = parser.parse_args()

    organize_raw_dataset(raw_dir=args.source, output_dir=args.target)


if __name__ == "__main__":
    main()