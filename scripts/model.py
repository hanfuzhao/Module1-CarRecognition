# Built with AI assistance (Claude). Uses torchvision ResNet50, scikit-learn, scikit-image.
"""
Model implementations for Stanford Cars classification: a naive baseline, a
classical HOG + linear SVM, and a transfer-learning deep model (frozen ResNet50
features with a trained MLP head). The deep model is the one we deploy; keeping
the backbone frozen means the saved artifact is only a few MB.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize as sk_resize
from PIL import Image

import torch
import torch.nn as nn

from scripts import data

PROCESSED = Path("data/processed")


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class NaiveBaseline:
    """Majority-class or uniform-random classifier (the performance floor)."""

    def __init__(self, strategy: str = "majority"):
        assert strategy in ("majority", "random")
        self.strategy = strategy
        self.most_common_class = None
        self.num_classes = None

    def fit(self, y_train: np.ndarray) -> "NaiveBaseline":
        self.num_classes = int(np.max(y_train)) + 1
        self.most_common_class = int(np.bincount(y_train).argmax())
        return self

    def predict(self, n: int) -> np.ndarray:
        if self.strategy == "majority":
            return np.full(n, self.most_common_class, dtype=np.int64)
        rng = np.random.default_rng(0)
        return rng.integers(0, self.num_classes, n)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)


class ClassicalModel:
    """HOG features fed to a linear SVM.

    The SVM is trained with SGD (hinge loss) rather than the dual LibLinear
    solver, which does not converge in reasonable time on 196 one-vs-rest
    classes; SGD reaches the same decision rule in seconds.
    """

    def __init__(self, img_size: int = 128):
        self.img_size = img_size
        self.clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", SGDClassifier(loss="hinge", alpha=1e-4, max_iter=30,
                                      tol=1e-3, random_state=0)),
            ]
        )

    def _hog(self, image: Image.Image) -> np.ndarray:
        arr = np.asarray(image.convert("RGB"))
        gray = rgb2gray(arr)
        gray = sk_resize(gray, (self.img_size, self.img_size), anti_aliasing=True)
        return hog(
            gray,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            feature_vector=True,
        )

    def extract(self, images: Sequence[Image.Image]) -> np.ndarray:
        """HOG feature vectors for a list of images."""
        return np.stack([self._hog(im) for im in images])

    def fit(self, features: np.ndarray, y: np.ndarray) -> "ClassicalModel":
        """Fit the linear SVM on precomputed HOG features."""
        self.clf.fit(features, y)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict labels from precomputed HOG features."""
        return self.clf.predict(features)

    def predict_proba_features(self, features: np.ndarray) -> np.ndarray:
        """Pseudo-probabilities from the SVM decision margins (softmax over
        one-vs-rest scores). The linear SVM is not calibrated, so these are
        relative confidences, used only for the app's top-5 display."""
        scores = np.asarray(self.clf.decision_function(features), dtype=np.float64)
        scores = scores - scores.max(axis=1, keepdims=True)
        e = np.exp(scores)
        return e / e.sum(axis=1, keepdims=True)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)


class _MLPHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 512, p: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class DeepModel:
    """Frozen ResNet50 features with a trained MLP head (transfer learning)."""

    FEAT_DIM = 2048

    def __init__(self, num_classes: int = 196, class_names: List[str] | None = None,
                 device: str | None = None):
        self.num_classes = num_classes
        self.class_names = class_names
        self.device = device or pick_device()
        self._backbone = None
        self._transform = None
        self.head = _MLPHead(self.FEAT_DIM, num_classes).to(self.device)

    def _ensure_backbone(self):
        if self._backbone is not None:
            return
        from torchvision.models import resnet50, ResNet50_Weights

        weights = ResNet50_Weights.IMAGENET1K_V2
        net = resnet50(weights=weights)
        net.fc = nn.Identity()
        net.eval().to(self.device)
        for p in net.parameters():
            p.requires_grad = False
        self._backbone = net
        self._transform = weights.transforms()

    @torch.no_grad()
    def extract_features(self, images: Sequence[Image.Image], batch_size: int = 64,
                         progress: bool = False) -> np.ndarray:
        self._ensure_backbone()
        feats = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            x = torch.stack([self._transform(im.convert("RGB")) for im in batch]).to(self.device)
            f = self._backbone(x)
            feats.append(f.cpu().numpy())
            if progress and (i // batch_size) % 20 == 0:
                print(f"    features {i + len(batch)}/{len(images)}", flush=True)
        return np.concatenate(feats, axis=0)

    def fit_features(self, feats: np.ndarray, y: np.ndarray, epochs: int = 60,
                     lr: float = 1e-3, batch_size: int = 256) -> "DeepModel":
        X = torch.tensor(feats, dtype=torch.float32)
        Y = torch.tensor(y, dtype=torch.long)
        ds = torch.utils.data.TensorDataset(X, Y)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

        opt = torch.optim.AdamW(self.head.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loss_fn = nn.CrossEntropyLoss()

        self.head.train()
        for ep in range(epochs):
            total = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.head(xb), yb)
                loss.backward()
                opt.step()
                total += loss.item() * len(xb)
            sched.step()
            if (ep + 1) % 10 == 0 or ep == 0:
                print(f"    epoch {ep + 1}/{epochs}  loss={total / len(ds):.4f}", flush=True)
        return self

    @torch.no_grad()
    def predict_proba_features(self, feats: np.ndarray) -> np.ndarray:
        self.head.eval()
        x = torch.tensor(feats, dtype=torch.float32, device=self.device)
        return torch.softmax(self.head(x), dim=1).cpu().numpy()

    @torch.no_grad()
    def predict_proba(self, images: Sequence[Image.Image]) -> np.ndarray:
        return self.predict_proba_features(self.extract_features(images))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "head_state": self.head.state_dict(),
                "num_classes": self.num_classes,
                "class_names": self.class_names,
                "feat_dim": self.FEAT_DIM,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "DeepModel":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(num_classes=ckpt["num_classes"], class_names=ckpt.get("class_names"),
                    device=device)
        model.head.load_state_dict(ckpt["head_state"])
        model.head.eval()
        return model


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {"accuracy": float(accuracy_score(y_true, y_pred))}


def topk_accuracy(y_true: np.ndarray, proba: np.ndarray, k: int = 5,
                  num_classes: int = 196) -> float:
    return float(top_k_accuracy_score(y_true, proba, k=k, labels=list(range(num_classes))))


# Feature generation over dataset splits, with caching to data/processed so
# re-runs and the robustness sweep are fast.
def deep_features(deep: DeepModel, split_name: str, limit: int | None = None):
    """Extract (and cache) ResNet50 features + labels for a split.

    `limit` caps the number of images, used to keep the robustness sweep over
    corruption splits fast (a sample per corruption is an adequate estimate and
    is noted in the report).
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
    feats, ys, done = [], [], 0
    t0 = time.time()
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
    feats, ys, done = [], [], 0
    t0 = time.time()
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
