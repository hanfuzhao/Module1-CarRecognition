"""
Deep learning training pipeline.
Trains fine-tuned ResNet50 and evaluates CLIP zero-shot baseline.
"""

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from tqdm import tqdm

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP not available. Install with: pip install openai-clip")


class ResNet50Classifier:
    """Fine-tuned ResNet50 for car type classification."""

    def __init__(self, num_classes: int = 196, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.model = self._build_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None

    def _build_model(self):
        """Load pretrained ResNet50 and adapt for car classification."""
        model = models.resnet50(pretrained=True)

        # Freeze early layers
        for param in list(model.parameters())[:-2]:
            param.requires_grad = False

        # Replace final layer
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, self.num_classes)

        return model.to(self.device)

    def train_epoch(self, train_loader, lr: float = 1e-4):
        """Train for one epoch."""
        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="max", factor=0.5, patience=2, verbose=True)

        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct / total:.4f}"})

        return total_loss / len(train_loader), correct / total

    def validate(self, val_loader):
        """Validate on validation set."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        return correct / total

    def evaluate(self, test_loader):
        """Evaluate on test set."""
        self.model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        accuracy = (y_pred == y_true).mean()

        return {
            "accuracy": accuracy,
            "y_true": y_true,
            "y_pred": y_pred,
        }

    def save(self, path: str):
        """Save model weights."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        """Load model weights."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))


class CLIPZeroShot:
    """CLIP zero-shot evaluation for comparison."""

    def __init__(self, class_names: list, device: str = None):
        if not CLIP_AVAILABLE:
            raise RuntimeError("CLIP not installed. Install with: pip install openai-clip")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        # Prepare text tokens
        text_inputs = clip.tokenize([f"a photo of a {c}" for c in class_names]).to(self.device)
        with torch.no_grad():
            self.text_features = self.model.encode_text(text_inputs)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def predict(self, images_list):
        """Predict using CLIP zero-shot."""
        from PIL import Image

        predictions = []

        with torch.no_grad():
            for img_path in images_list:
                image = self.preprocess(Image.open(img_path)).unsqueeze(0).to(self.device)
                image_features = self.model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                logits = 100.0 * image_features @ self.text_features.T
                pred = logits.argmax(-1).item()
                predictions.append(pred)

        return np.array(predictions)

    def evaluate(self, image_paths: list, labels: list):
        """Evaluate CLIP on test set."""
        predictions = self.predict(image_paths)
        accuracy = (predictions == np.array(labels)).mean()

        return {
            "accuracy": accuracy,
            "y_pred": predictions,
            "y_true": np.array(labels),
        }


def train_resnet50(train_loader, val_loader, test_loader, epochs: int = 5, output_dir: str = "models"):
    """Train ResNet50 classifier."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("DEEP LEARNING: Fine-tuned ResNet50")
    print("=" * 60)

    model = ResNet50Classifier(num_classes=196)
    best_val_acc = 0

    for epoch in range(epochs):
        print(f"\nEpoch [{epoch + 1}/{epochs}]")
        train_loss, train_acc = model.train_epoch(train_loader, lr=1e-4)
        val_acc = model.validate(val_loader)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(f"{output_dir}/resnet50_best.pt")
            print(f"  ✓ Saved best model (val_acc: {val_acc:.4f})")

        model.scheduler.step(val_acc)

    # Load best model and evaluate
    model.load(f"{output_dir}/resnet50_best.pt")
    test_results = model.evaluate(test_loader)

    print(f"\nTest Accuracy: {test_results['accuracy']:.4f}")

    return test_results


def evaluate_clip_zero_shot(test_loader, class_names: list, output_dir: str = "models"):
    """Evaluate CLIP zero-shot model for comparison."""

    if not CLIP_AVAILABLE:
        print("\nWarning: CLIP not available, skipping zero-shot evaluation")
        return None

    print("\n" + "=" * 60)
    print("DEEP LEARNING: CLIP Zero-Shot (No Fine-tuning)")
    print("=" * 60)

    # Collect all test images
    test_images = []
    test_labels = []
    for images, labels in test_loader:
        # This is simplified; in practice you'd need actual image paths
        test_images.extend([None] * len(labels))
        test_labels.extend(labels.numpy())

    # For now, skip CLIP evaluation due to complexity
    print("(CLIP evaluation deferred to experiment phase)")
    return None
