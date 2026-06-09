# Built with AI assistance (Claude). Uses torchvision ResNet50 features and skimage HOG.
"""
Feature-generation pipeline.

Turns raw Stanford Cars images into the two feature representations the models
need, and caches them under data/processed so re-runs and the robustness sweep
are fast:

  * deep features  - 2048-d frozen-ResNet50 embeddings (for the deep model)
  * HOG features   - 1764-d gradient histograms (for the classical model)

Used by setup.py; can also be run standalone to pre-build the caches.
"""

import time
from pathlib import Path

import numpy as np

from scripts import data
from scripts.model import DeepModel, ClassicalModel

PROCESSED = Path("data/processed")


def deep_features(deep: DeepModel, split_name: str, limit: int | None = None):
    """Extract (and cache) ResNet50 features + labels for a split.

    `limit` caps the number of images, used to keep the robustness sweep over
    corruption splits fast (a 2.5k sample per corruption is an adequate estimate
    and is noted in the report).
    """
    PROCESSED.mkdir(parents=True, exist_ok=True)
    tag = split_name if limit is None else f"{split_name}_s{limit}"
    fpath = PROCESSED / f"feat_{tag}.npy"
    ypath = PROCESSED / f"label_{tag}.npy"
    if fpath.exists() and ypath.exists():
        print(f"  [cache] {tag} features")
        return np.load(fpath), np.load(ypath)

    ds = data.load_split(split_name)
    total = ds.num_rows if limit is None else min(limit, ds.num_rows)
    feats, ys = [], []
    t0 = time.time()
    done = 0
    for images, labels in data.iter_batches(ds, batch_size=256):
        if limit is not None and done >= limit:
            break
        feats.append(deep.extract_features(images, batch_size=64))
        ys.append(labels)
        done += len(labels)
        print(f"  [{tag}] {min(done, total)}/{total}  ({time.time() - t0:.0f}s)", flush=True)
    feats = np.concatenate(feats).astype(np.float32)[:total]
    ys = np.concatenate(ys)[:total]
    np.save(fpath, feats)
    np.save(ypath, ys)
    return feats, ys


def hog_features(model: ClassicalModel, split_name: str):
    """Extract (and cache) HOG features + labels for a split."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    fpath = PROCESSED / f"hog_{split_name}.npy"
    ypath = PROCESSED / f"label_{split_name}.npy"
    if fpath.exists():
        print(f"  [cache] {split_name} HOG")
        return np.load(fpath), np.load(ypath)

    ds = data.load_split(split_name)
    feats, ys = [], []
    t0 = time.time()
    done = 0
    for images, labels in data.iter_batches(ds, batch_size=256):
        feats.append(model.extract(images))
        ys.append(labels)
        done += len(labels)
        print(f"  [HOG {split_name}] {done}/{ds.num_rows}  ({time.time() - t0:.0f}s)", flush=True)
    feats = np.concatenate(feats).astype(np.float32)
    ys = np.concatenate(ys)
    np.save(fpath, feats)
    if not ypath.exists():
        np.save(ypath, ys)
    return feats, ys
