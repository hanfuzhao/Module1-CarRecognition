# Built with AI assistance (Claude). Uses matplotlib and Hugging Face datasets.
"""
Exploratory data analysis for Stanford Cars.

Reports class-distribution statistics (to justify the choice of robustness over
long-tail as the primary experiment) and saves a sample-image montage.

Run:  python scripts/eda.py
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts import data

OUTPUTS = Path("data/outputs")


def class_distribution(train_ds):
    """Return per-class training counts and summary statistics."""
    labels = data.all_labels(train_ds)
    counts = np.bincount(labels, minlength=len(train_ds.features["label"].names))
    stats = {
        "num_classes": int(len(counts)),
        "total": int(counts.sum()),
        "min_per_class": int(counts.min()),
        "max_per_class": int(counts.max()),
        "mean_per_class": float(counts.mean()),
        "std_per_class": float(counts.std()),
    }
    return counts, stats


def plot_distribution(counts):
    plt.figure(figsize=(7, 4))
    plt.hist(counts, bins=20, edgecolor="black", alpha=0.8)
    plt.axvline(np.median(counts), color="r", linestyle="--",
                label=f"median={np.median(counts):.0f}")
    plt.xlabel("Training images per class")
    plt.ylabel("Number of classes")
    plt.title("Stanford Cars class distribution (train)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS / "class_distribution.png", dpi=120)
    plt.close()


def sample_montage(train_ds, class_names, n=8):
    rows = train_ds[0:n]
    fig, axes = plt.subplots(2, n // 2, figsize=(12, 6))
    for ax, im, lbl in zip(axes.ravel(), rows["image"], rows["label"]):
        ax.imshow(im)
        ax.set_title(class_names[lbl], fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "sample_images.png", dpi=110)
    plt.close()


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    train_ds = data.load_split("train")
    class_names = data.get_class_names(train_ds)
    counts, stats = class_distribution(train_ds)

    print("=== Stanford Cars EDA ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  imbalance ratio (max/min): {stats['max_per_class'] / stats['min_per_class']:.2f}")
    print("  -> distribution is near-balanced, so robustness (not long-tail) is the primary experiment")

    plot_distribution(counts)
    sample_montage(train_ds, class_names)
    print(f"\nSaved plots to {OUTPUTS}/")


if __name__ == "__main__":
    main()
