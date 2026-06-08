"""
Data access for the Stanford Cars dataset.

Source: tanganke/stanford_cars on the Hugging Face Hub
  https://huggingface.co/datasets/tanganke/stanford_cars
This mirror provides the canonical train/test splits (196 classes) plus
seven corruption splits (contrast, gaussian_noise, impulse_noise,
jpeg_compression, motion_blur, pixelate, spatter) used for the
robustness experiment.

All splits expose two columns: `image` (PIL.Image) and `label` (int 0-195).
"""

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from datasets import load_dataset

HF_DATASET = "tanganke/stanford_cars"
CLEAN_SPLITS = ("train", "test")
CORRUPTION_SPLITS = (
    "gaussian_noise",
    "motion_blur",
    "jpeg_compression",
    "pixelate",
    "contrast",
    "impulse_noise",
    "spatter",
)
METADATA_PATH = Path("data/raw/stanford-cars/metadata.json")


def load_split(split: str):
    """Load a single split as a Hugging Face Dataset (cached after first run)."""
    return load_dataset(HF_DATASET, split=split)


def get_class_names(split=None) -> List[str]:
    """Return the 196 human-readable class names (e.g. 'Acura RL Sedan 2012')."""
    ds = split if split is not None else load_split("train")
    return list(ds.features["label"].names)


def split_to_arrays(ds) -> Tuple[list, np.ndarray]:
    """Convert a split into (list_of_PIL_images, label_array). Small splits only."""
    images = [img.convert("RGB") for img in ds["image"]]
    labels = np.array(ds["label"], dtype=np.int64)
    return images, labels


def iter_batches(ds, batch_size: int = 256):
    """
    Stream a split in batches to keep memory bounded (full splits are ~9 GB of
    decoded pixels). Yields (list_of_PIL_images, label_array) per batch.
    """
    n = ds.num_rows
    for start in range(0, n, batch_size):
        rows = ds[start : start + batch_size]
        images = [im.convert("RGB") for im in rows["image"]]
        labels = np.array(rows["label"], dtype=np.int64)
        yield images, labels


def all_labels(ds) -> np.ndarray:
    """Return just the label column (no image decoding)."""
    return np.array(ds["label"], dtype=np.int64)


def write_metadata(class_names: List[str], n_train: int, n_test: int) -> None:
    """Persist dataset metadata for the app and EDA to consume."""
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": HF_DATASET,
        "num_classes": len(class_names),
        "num_train": n_train,
        "num_test": n_test,
        "class_names": class_names,
        "corruption_splits": list(CORRUPTION_SPLITS),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(payload, f, indent=2)


def load_metadata() -> dict:
    """Load persisted metadata, or raise if make_dataset has not been run."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"{METADATA_PATH} not found. Run `python scripts/make_dataset.py` first."
        )
    with open(METADATA_PATH) as f:
        return json.load(f)
