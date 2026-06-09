# Built with AI assistance (Claude). Data: Stanford Cars via HF tanganke/stanford_cars.
"""
Get the data.

Default run downloads Stanford Cars and writes every image to disk under
data/raw/stanford-cars/{train,test}/<label>/, so the raw directory holds the
full ~16,185 images locally (the bulk is git-ignored; only metadata and a small
committed sample live in the repo).

Usage:
  python scripts/make_dataset.py            # write the full train+test images
  python scripts/make_dataset.py --sample   # write one image per class to data/raw/sample
"""

import os
import sys
import re

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# allow running both as `python scripts/make_dataset.py` and `python -m scripts.make_dataset`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from scripts import data

RAW_ROOT = Path("data/raw/stanford-cars")
SAMPLE_ROOT = Path("data/raw/sample")


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def save_full(split_name: str) -> int:
    """Write every image of a split to data/raw/stanford-cars/<split>/<label>/."""
    ds = data.load_split(split_name)
    out_root = RAW_ROOT / split_name
    count = 0
    for images, labels in data.iter_batches(ds, batch_size=256):
        for img, lbl in zip(images, labels):
            class_dir = out_root / f"{int(lbl):03d}"
            class_dir.mkdir(parents=True, exist_ok=True)
            img.save(class_dir / f"{count:06d}.jpg", quality=90)
            count += 1
        print(f"  [{split_name}] {count}/{ds.num_rows}", flush=True)
    return count


def save_sample(class_names) -> int:
    """Write one representative image per class to data/raw/sample/ (committed)."""
    ds = data.load_split("train")
    SAMPLE_ROOT.mkdir(parents=True, exist_ok=True)
    seen = set()
    for images, labels in data.iter_batches(ds, batch_size=256):
        for img, lbl in zip(images, labels):
            lbl = int(lbl)
            if lbl in seen:
                continue
            seen.add(lbl)
            img.save(SAMPLE_ROOT / f"{lbl:03d}_{_safe(class_names[lbl])}.jpg", quality=85)
        if len(seen) == len(class_names):
            break
    return len(seen)


def main():
    print("Loading Stanford Cars from Hugging Face (cached after first run)...")
    train = data.load_split("train")
    test = data.load_split("test")
    class_names = data.get_class_names(train)
    data.write_metadata(class_names, train.num_rows, test.num_rows)
    print(f"  classes: {len(class_names)} | train: {train.num_rows} | test: {test.num_rows}")

    if "--sample" in sys.argv:
        n = save_sample(class_names)
        print(f"\nWrote {n} sample images to {SAMPLE_ROOT}/")
        return

    n_train = save_full("train")
    n_test = save_full("test")
    print(f"\nWrote {n_train + n_test} images to {RAW_ROOT}/ (train={n_train}, test={n_test})")
    print(f"Metadata: {data.METADATA_PATH}")


if __name__ == "__main__":
    main()
