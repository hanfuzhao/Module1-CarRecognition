"""
Model implementations for Stanford Cars classification.
Three approaches: Naive Baseline, Classical ML, and Deep Learning.
"""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skimage import feature
from PIL import Image
import torch
import torchvision.transforms as transforms


class BaseModel(ABC):
    """Abstract base class for all models."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.num_classes = None

    @abstractmethod
    def train(self, X_train, y_train):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X):
        """Make predictions."""
        pass

    def evaluate(self, X_test, y_test):
        """Evaluate model on test set."""
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        return {
            "accuracy": acc,
            "y_pred": y_pred,
            "report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }

    def save(self, path: str):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        """Load model from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)


class NaiveBaseline(BaseModel):
    """Naive baseline models: Majority class and Random classifier."""

    def __init__(self, strategy: str = "majority"):
        super().__init__("NaiveBaseline")
        assert strategy in ["majority", "random"], f"Unknown strategy: {strategy}"
        self.strategy = strategy
        self.most_common_class = None

    def train(self, X_train, y_train):
        """Train baseline model."""
        self.num_classes = len(np.unique(y_train))
        if self.strategy == "majority":
            self.most_common_class = np.bincount(y_train).argmax()
        return self

    def predict(self, X):
        """Make predictions."""
        n_samples = len(X) if isinstance(X, (list, np.ndarray)) else X.shape[0]

        if self.strategy == "majority":
            return np.full(n_samples, self.most_common_class)
        else:  # random
            return np.random.randint(0, self.num_classes, n_samples)


class ClassicalMLModel(BaseModel):
    """Classical ML model using HOG features + SVM."""

    def __init__(self, model_type: str = "svm", img_size: int = 128):
        super().__init__("ClassicalML")
        self.model_type = model_type
        self.img_size = img_size
        self.feature_extractor = self._build_feature_extractor()
        self.classifier = self._build_classifier()

    def _build_feature_extractor(self):
        """Build HOG feature extraction pipeline."""
        return lambda img: self._extract_hog_features(img)

    @staticmethod
    def _extract_hog_features(img_path):
        """Extract HOG features from image."""
        img = Image.open(img_path).convert("L")
        img = np.array(img.resize((128, 128)))
        hog_features = feature.hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        return hog_features

    def _build_classifier(self):
        """Build classifier pipeline."""
        if self.model_type == "svm":
            return Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", C=10, gamma="scale"))])
        else:  # random_forest
            return Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=100, n_jobs=-1))])

    def train(self, X_train, y_train):
        """Train classical ML model. X_train should be list of image paths."""
        print(f"Extracting HOG features from {len(X_train)} training images...")
        X_features = np.array([self._extract_hog_features(img_path) for img_path in X_train])

        print(f"Training {self.model_type} classifier...")
        self.classifier.fit(X_features, y_train)
        self.num_classes = len(np.unique(y_train))
        return self

    def predict(self, X):
        """Make predictions. X can be image paths or feature arrays."""
        if isinstance(X[0], str):
            X = np.array([self._extract_hog_features(img_path) for img_path in X])

        return self.classifier.predict(X)

    def predict_proba(self, X):
        """Predict probabilities."""
        if isinstance(X[0], str):
            X = np.array([self._extract_hog_features(img_path) for img_path in X])

        if hasattr(self.classifier.named_steps["svm" if self.model_type == "svm" else "rf"], "predict_proba"):
            return self.classifier.predict_proba(X)
        return None


class DeepLearningModel(BaseModel):
    """Deep learning model using pretrained ResNet50 backbone."""

    def __init__(self, num_classes: int = 196, pretrained: bool = True, device: str = None):
        super().__init__("DeepLearning")
        self.num_classes = num_classes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model(pretrained)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _build_model(self, pretrained: bool):
        """Build ResNet50 backbone with custom classifier."""
        from torchvision.models import resnet50

        model = resnet50(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, self.num_classes)
        return model.to(self.device)

    def train(self, train_loader, val_loader=None, epochs: int = 10, lr: float = 1e-3):
        """Train the deep learning model."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.4f}")

        return self

    def predict(self, X):
        """Make predictions on batch of images."""
        self.model.eval()
        with torch.no_grad():
            if isinstance(X, list):
                images = torch.stack([self.transform(Image.open(img_path)) for img_path in X])
            else:
                images = X

            images = images.to(self.device)
            outputs = self.model(images)
            _, predictions = torch.max(outputs, 1)

        return predictions.cpu().numpy()

    def predict_proba(self, X):
        """Predict probabilities on batch of images."""
        self.model.eval()
        with torch.no_grad():
            if isinstance(X, list):
                images = torch.stack([self.transform(Image.open(img_path)) for img_path in X])
            else:
                images = X

            images = images.to(self.device)
            outputs = self.model(images)
            probs = torch.softmax(outputs, dim=1)

        return probs.cpu().numpy()

    def save(self, path: str):
        """Save model weights."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    @classmethod
    def load(cls, path: str, num_classes: int = 196):
        """Load model weights."""
        model = cls(num_classes=num_classes)
        model.model.load_state_dict(torch.load(path, map_location=model.device))
        return model
