"""
Interactive Flask application for car type recognition.
Upload image → Get prediction + confidence scores + guidance.

Run: python main.py
Access: http://localhost:5000
"""

import os
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from flask import Flask, render_template, request, jsonify, send_from_directory

from scripts.model import NaiveBaseline, ClassicalMLModel, DeepLearningModel


class PredictionService:
    """Inference service combining all three models."""

    def __init__(self, model_dir: str = "models", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        self.models = {}
        self.class_names = self._load_class_names()
        self.load_models()

    def _load_class_names(self):
        """Load class names from metadata."""
        metadata_path = Path("data/raw/stanford-cars/metadata.json")
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                return metadata["class_names"]
        return [f"Class {i}" for i in range(196)]

    def load_models(self):
        """Load trained models."""
        try:
            naive_path = f"{self.model_dir}/naive_majority.pkl"
            if Path(naive_path).exists():
                self.models["naive"] = NaiveBaseline.load(naive_path)
        except Exception as e:
            print(f"Warning: Could not load naive baseline: {e}")

        try:
            classical_path = f"{self.model_dir}/classical_svm.pkl"
            if Path(classical_path).exists():
                self.models["classical"] = ClassicalMLModel.load(classical_path)
        except Exception as e:
            print(f"Warning: Could not load classical model: {e}")

        try:
            dl_path = f"{self.model_dir}/resnet50_best.pt"
            if Path(dl_path).exists():
                self.models["deep_learning"] = DeepLearningModel.load(dl_path, num_classes=196)
        except Exception as e:
            print(f"Warning: Could not load DL model: {e}")

    def predict(self, image_path: str, confidence_threshold: float = 0.5):
        """Run inference and return results."""
        image = Image.open(image_path).convert("RGB")

        results = {}

        # Deep Learning model (primary)
        if "deep_learning" in self.models:
            try:
                probs = self.models["deep_learning"].predict_proba([image_path])[0]
                top_5_idx = np.argsort(probs)[::-1][:5]
                top_5_probs = probs[top_5_idx]

                pred_class = top_5_idx[0]
                confidence = float(top_5_probs[0])

                results["primary"] = {
                    "model": "ResNet50 (Fine-tuned)",
                    "prediction": self.class_names[pred_class],
                    "class_id": int(pred_class),
                    "confidence": confidence,
                    "top_5": [
                        {"class": self.class_names[idx], "confidence": float(prob)} for idx, prob in zip(top_5_idx, top_5_probs)
                    ],
                }

                # Confidence feedback
                if confidence < confidence_threshold:
                    results["feedback"] = {
                        "level": "low_confidence",
                        "message": f"Confidence is {confidence:.1%}. Try a different angle or lighting for better accuracy.",
                        "suggestion": "📸 Try retaking the photo with better lighting or a different car angle.",
                    }
                else:
                    results["feedback"] = {
                        "level": "confident",
                        "message": f"High confidence ({confidence:.1%}). This is a {results['primary']['prediction']}.",
                    }
            except Exception as e:
                results["error"] = f"Deep learning inference failed: {str(e)}"

        # Classical model (backup)
        if "classical" in self.models and "primary" not in results:
            try:
                pred = self.models["classical"].predict([image_path])[0]
                results["backup"] = {
                    "model": "Classical ML (SVM)",
                    "prediction": self.class_names[pred],
                    "class_id": int(pred),
                }
            except Exception as e:
                print(f"Classical inference failed: {e}")

        return results


app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size
app.config["UPLOAD_FOLDER"] = Path("static/uploads")
app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)

predictor = PredictionService()


@app.route("/")
def index():
    """Home page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload and prediction."""
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Save uploaded file
        filename = f"upload_{int(np.random.random() * 1e9)}.jpg"
        filepath = app.config["UPLOAD_FOLDER"] / filename
        file.save(filepath)

        # Run prediction
        results = predictor.predict(str(filepath), confidence_threshold=0.6)

        # Clean up (optional: keep for debugging)
        # filepath.unlink()

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/upload/<filename>")
def get_upload(filename):
    """Serve uploaded images."""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/health")
def health():
    """Health check."""
    return jsonify({"status": "ok", "models_loaded": list(predictor.models.keys())})


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large (max 16MB)"}), 413


if __name__ == "__main__":
    print("=" * 60)
    print("Car Type Recognition - Interactive App")
    print("=" * 60)
    print(f"Models loaded: {list(predictor.models.keys())}")
    print(f"Available classes: {len(predictor.class_names)}")
    print("\nStarting Flask server...")
    print("Access at: http://localhost:5000")
    print("=" * 60)

    app.run(debug=False, host="0.0.0.0", port=5000)
