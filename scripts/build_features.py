"""
Build data loaders and prepare datasets for training.
Handles train/test split and creates PyTorch DataLoaders.
"""

import json
from pathlib import Path
from typing import Tuple, List
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


class StanfordCarsDataset(Dataset):
    """PyTorch Dataset for Stanford Cars."""

    def __init__(self, img_dir: str, split: str = "train", transform=None):
        """
        Args:
            img_dir: Root directory containing train/test subdirectories
            split: 'train' or 'test'
            transform: Optional image transforms
        """
        self.split = split
        self.transform = transform
        self.images = []
        self.labels = []

        split_dir = Path(img_dir) / split
        class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])

        for class_id, class_dir in enumerate(class_dirs):
            for img_path in sorted(class_dir.glob("*.jpg")):
                self.images.append(str(img_path))
                self.labels.append(int(class_dir.name))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_distribution(self):
        """Return class distribution statistics."""
        counts = np.bincount(self.labels)
        return {
            "num_classes": len(counts),
            "num_samples": len(self),
            "min_samples": int(counts.min()),
            "max_samples": int(counts.max()),
            "mean_samples": float(counts.mean()),
            "class_counts": counts.tolist(),
        }


def create_train_transforms():
    """Augmentation transforms for training."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def create_val_transforms():
    """Validation transforms (no augmentation)."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_dataloaders(
    data_dir: str = "data/raw/stanford-cars",
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test dataloaders.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_transform = create_train_transforms()
    val_transform = create_val_transforms()

    # Load train dataset and split into train/val
    full_train_dataset = StanfordCarsDataset(data_dir, split="train", transform=train_transform)
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size

    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform  # Switch to val transforms

    test_dataset = StanfordCarsDataset(data_dir, split="test", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    print(f"Classes: {full_train_dataset.get_class_distribution()['num_classes']}")

    return train_loader, val_loader, test_loader


def get_image_paths(data_dir: str = "data/raw/stanford-cars", split: str = "train") -> Tuple[List[str], List[int]]:
    """Get lists of image paths and labels for classical ML models."""
    img_paths = []
    labels = []

    split_dir = Path(data_dir) / split
    class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])

    for class_id, class_dir in enumerate(class_dirs):
        for img_path in sorted(class_dir.glob("*.jpg")):
            img_paths.append(str(img_path))
            labels.append(int(class_dir.name))

    return img_paths, np.array(labels)
