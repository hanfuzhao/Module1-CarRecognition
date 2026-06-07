"""
Download and prepare Stanford Cars dataset.
Source: https://ai.stanford.edu/~jkrause/cars/car_dataset.html
Mirrors: Hugging Face, Kaggle
"""

import os
import sys
from pathlib import Path
from datasets import load_dataset
from PIL import Image
import json


def download_stanford_cars(output_dir: str = "data/raw/stanford-cars"):
    """Download Stanford Cars dataset from Hugging Face."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Downloading Stanford Cars dataset from Hugging Face...")
    dataset = load_dataset("gvlproject/stanford-cars")

    # Save train split
    train_dir = output_path / "train"
    train_dir.mkdir(exist_ok=True)

    print(f"Processing {len(dataset['train'])} training images...")
    for idx, example in enumerate(dataset["train"]):
        image = example["image"]
        label = example["label"]
        class_dir = train_dir / str(label)
        class_dir.mkdir(exist_ok=True)

        img_path = class_dir / f"{idx:06d}.jpg"
        image.save(img_path)

        if (idx + 1) % 1000 == 0:
            print(f"  Saved {idx + 1} training images")

    # Save test split
    test_dir = output_path / "test"
    test_dir.mkdir(exist_ok=True)

    print(f"Processing {len(dataset['test'])} test images...")
    for idx, example in enumerate(dataset["test"]):
        image = example["image"]
        label = example["label"]
        class_dir = test_dir / str(label)
        class_dir.mkdir(exist_ok=True)

        img_path = class_dir / f"{idx:06d}.jpg"
        image.save(img_path)

        if (idx + 1) % 1000 == 0:
            print(f"  Saved {idx + 1} test images")

    # Save metadata
    metadata = {
        "dataset": "Stanford Cars",
        "num_classes": dataset["train"].features["label"].num_classes,
        "num_train": len(dataset["train"]),
        "num_test": len(dataset["test"]),
        "class_names": dataset["train"].features["label"].names,
    }

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Dataset saved to {output_path}")
    print(f"  Train: {metadata['num_train']} images ({metadata['num_classes']} classes)")
    print(f"  Test:  {metadata['num_test']} images")

    return output_path


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/stanford-cars"
    download_stanford_cars(output_dir)
