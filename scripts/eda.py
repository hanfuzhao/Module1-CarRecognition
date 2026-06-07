"""
Exploratory Data Analysis for Stanford Cars dataset.
Generates statistics and visualizations.
"""

import json
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_metadata(data_path: str = "data/raw/stanford-cars"):
    """Load dataset metadata."""
    metadata_path = Path(data_path) / "metadata.json"
    with open(metadata_path, "r") as f:
        return json.load(f)


def analyze_class_distribution(data_path: str = "data/raw/stanford-cars"):
    """Analyze class distribution."""
    metadata = load_metadata(data_path)
    data_root = Path(data_path)

    train_counts = Counter()
    train_dir = data_root / "train"
    for class_dir in sorted(train_dir.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.jpg")))
            train_counts[int(class_dir.name)] = count

    print("\n=== Class Distribution (Train) ===")
    print(f"Classes: {metadata['num_classes']}")
    print(f"Total images: {metadata['num_train']}")
    print(f"Mean per class: {metadata['num_train'] / metadata['num_classes']:.1f}")
    print(f"Max per class: {max(train_counts.values())}")
    print(f"Min per class: {min(train_counts.values())}")

    # Identify head vs tail
    median = np.median(list(train_counts.values()))
    head_classes = sum(1 for c in train_counts.values() if c > median)
    tail_classes = sum(1 for c in train_counts.values() if c <= median)
    print(f"\nHead classes (>{median:.0f}): {head_classes}")
    print(f"Tail classes (<={median:.0f}): {tail_classes}")

    return train_counts, metadata


def sample_images(data_path: str = "data/raw/stanford-cars", num_samples: int = 5):
    """Visualize sample images from different classes."""
    metadata = load_metadata(data_path)
    data_root = Path(data_path)
    train_dir = data_root / "train"

    class_dirs = sorted(train_dir.iterdir())
    sample_classes = np.random.choice(len(class_dirs), min(num_samples, len(class_dirs)), replace=False)

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    if num_samples == 1:
        axes = [axes]

    for i, class_idx in enumerate(sample_classes):
        class_dir = class_dirs[class_idx]
        img_file = list(class_dir.glob("*.jpg"))[0]
        img = Image.open(img_file)

        axes[i].imshow(img)
        axes[i].set_title(f"Class {class_idx}\n{metadata['class_names'][class_idx]}", fontsize=9)
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("data/outputs/sample_images.png", dpi=100, bbox_inches="tight")
    print(f"✓ Sample images saved to data/outputs/sample_images.png")


def plot_class_distribution(train_counts, metadata):
    """Plot class distribution histogram."""
    counts = sorted(train_counts.values())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    axes[0].hist(counts, bins=20, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Samples per class")
    axes[0].set_ylabel("Number of classes")
    axes[0].set_title("Class Distribution (Train)")
    axes[0].axvline(np.median(counts), color="r", linestyle="--", label=f"Median: {np.median(counts):.0f}")
    axes[0].legend()

    # Sorted bar chart (top 20 and bottom 20)
    sorted_counts = sorted(enumerate(train_counts.values()), key=lambda x: x[1], reverse=True)
    top_20 = sorted_counts[:20]
    bottom_20 = sorted_counts[-20:]

    x_top = range(len(top_20))
    axes[1].bar(x_top, [c for _, c in top_20], alpha=0.7, label="Top 20")
    axes[1].set_xlabel("Class rank")
    axes[1].set_ylabel("Samples")
    axes[1].set_title("Top 20 and Bottom 20 Classes")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("data/outputs/class_distribution.png", dpi=100, bbox_inches="tight")
    print(f"✓ Class distribution plot saved to data/outputs/class_distribution.png")


def main(data_path: str = "data/raw/stanford-cars"):
    """Run full EDA."""
    print("=== Stanford Cars EDA ===")

    train_counts, metadata = analyze_class_distribution(data_path)
    sample_images(data_path)
    plot_class_distribution(train_counts, metadata)

    print("\n✓ EDA complete!")


if __name__ == "__main__":
    main()
