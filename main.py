# Built with AI assistance (Claude). Uses Flask and torch/torchvision.
"""
Interactive web app for car-type recognition (inference only).

Upload a car photo and pick one of the three trained models (naive baseline,
classical HOG + SVM, or the deep ResNet50 + MLP). The app returns the top
predictions with confidence and a confidence-aware note, so the difference in
model performance is visible side by side.

Run:  python main.py    then open http://localhost:5000
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import io
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify

from scripts.model import DeepModel

MODELS_DIR = Path("models")
DEEP_PATH = MODELS_DIR / "deep_resnet50_features_mlp_head.pt"
CLASSICAL_PATH = MODELS_DIR / "classical_hog_linear_svm.pkl"
NAIVE_PATH = MODELS_DIR / "naive_baseline_majority_class.pkl"
METADATA_PATH = Path("data/raw/stanford-cars/metadata.json")
CONFIDENCE_THRESHOLD = 0.40  # below this we ask the user for a clearer photo

# Headline test-set accuracy for each model (from data/outputs/metrics.json).
MODELS = {
    "deep": {"label": "Deep: ResNet50 features + MLP", "accuracy": 0.5216},
    "classical": {"label": "Classical: HOG + linear SVM", "accuracy": 0.0441},
    "naive": {"label": "Naive: majority class", "accuracy": 0.0085},
}


class PredictionService:
    """Loads the three trained models once and serves predictions."""

    def __init__(self):
        self.deep = DeepModel.load(str(DEEP_PATH))
        self.deep._ensure_backbone()  # warm up the frozen ResNet50 backbone
        self.classical = self._load_pickle(CLASSICAL_PATH)
        self.naive = self._load_pickle(NAIVE_PATH)
        self.class_names = self.deep.class_names or self._fallback_names()

    @staticmethod
    def _load_pickle(path):
        if Path(path).exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def _fallback_names(self):
        if METADATA_PATH.exists():
            return json.loads(METADATA_PATH.read_text())["class_names"]
        return [f"Class {i}" for i in range(self.deep.num_classes)]

    def _deep_result(self, base, image, topk):
        proba = self.deep.predict_proba([image])[0]
        order = np.argsort(proba)[::-1][:topk]
        peak = float(proba[order[0]]) or 1.0
        top = [{
            "label": self.class_names[i],
            "pct": f"{proba[i] * 100:.1f}%",
            "bar": max(3.0, float(proba[i]) / peak * 100),
        } for i in order]
        conf = float(proba[order[0]])
        if conf < CONFIDENCE_THRESHOLD:
            fb = {"level": "low_confidence",
                  "message": f"Only {conf:.0%} confident. The photo may be unclear, cropped, "
                             f"or an angle the model has not seen.",
                  "suggestion": "Try a clearer, well-lit photo of the whole car (3/4 front view)."}
        else:
            fb = {"level": "confident", "message": f"{conf:.0%} confident this is a {top[0]['label']}."}
        base.update({"prediction": top[0]["label"], "confidence": conf, "top_k": top, "feedback": fb})
        return base

    def _classical_result(self, base, image, topk):
        feats = self.classical.extract([image])
        scores = np.asarray(self.classical.clf.decision_function(feats))[0]
        order = np.argsort(scores)[::-1][:topk]
        margins = scores[order]
        span = float(margins[0] - margins[-1]) or 1.0
        top = [{
            "label": self.class_names[i],
            "pct": None,  # SVM margins are not calibrated probabilities
            "bar": max(3.0, float(margins[k] - margins[-1]) / span * 100),
        } for k, i in enumerate(order)]
        base.update({
            "prediction": self.class_names[order[0]],
            "confidence": None,
            "top_k": top,
            "note": "Linear SVM returns a ranked class list, not calibrated probabilities.",
            "feedback": {"level": "low_confidence",
                         "message": "Classical HOG + SVM is a weak baseline (4.4% test accuracy); "
                                    "read its output as a ranking, not a confident answer."},
        })
        return base

    def _naive_result(self, base):
        cls = int(self.naive.most_common_class)
        base.update({
            "prediction": self.class_names[cls],
            "confidence": None,
            "top_k": [],
            "feedback": {"level": "low_confidence",
                         "message": "The naive baseline always predicts the single most common "
                                    "training class and ignores the image entirely.",
                         "suggestion": "It is only a floor to compare the other models against."},
        })
        return base

    def predict(self, image: Image.Image, model_key: str = "deep", topk: int = 5) -> dict:
        if model_key not in MODELS:
            model_key = "deep"
        meta = MODELS[model_key]
        base = {"model": model_key, "model_label": meta["label"], "model_accuracy": meta["accuracy"]}
        if model_key == "naive":
            return self._naive_result(base)
        if model_key == "classical":
            return self._classical_result(base, image, topk)
        return self._deep_result(base, image, topk)


app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

_service = None


def get_service() -> PredictionService:
    global _service
    if _service is None:
        _service = PredictionService()
    return _service


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    ok = DEEP_PATH.exists()
    return jsonify({"status": "ok" if ok else "model_missing",
                    "num_classes": get_service().deep.num_classes if ok else 0})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files or request.files["image"].filename == "":
        return jsonify({"error": "No image provided"}), 400
    try:
        raw = request.files["image"].read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return jsonify({"error": "Could not read image file"}), 400

    model_key = request.form.get("model", "deep")
    try:
        return jsonify(get_service().predict(image, model_key))
    except Exception as e:  # pragma: no cover - defensive
        return jsonify({"error": f"Inference failed: {e}"}), 500


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "File too large (max 16MB)"}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("Loading models and warming up backbone...")
    get_service()
    print(f"Ready. Open http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
