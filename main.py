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

    def _format(self, base, proba, topk):
        """Fill prediction / confidence / top_k from a probability vector."""
        order = np.argsort(proba)[::-1][:topk]
        peak = float(proba[order[0]]) or 1.0
        base["top_k"] = [{
            "label": self.class_names[i],
            "pct": f"{proba[i] * 100:.1f}%",
            "bar": max(3.0, float(proba[i]) / peak * 100),
        } for i in order]
        base["prediction"] = self.class_names[order[0]]
        base["confidence"] = float(proba[order[0]])
        return base

    def _classical_proba(self, image):
        """Approximate probabilities: softmax over standardized SVM margins."""
        feats = self.classical.extract([image])
        s = np.asarray(self.classical.clf.decision_function(feats))[0].astype(np.float64)
        z = (s - s.mean()) / (s.std() + 1e-9)
        e = np.exp(z - z.max())
        return e / e.sum()

    def predict(self, image: Image.Image, model_key: str = "deep", topk: int = 5) -> dict:
        if model_key not in MODELS:
            model_key = "deep"
        meta = MODELS[model_key]
        base = {"model": model_key, "model_label": meta["label"], "model_accuracy": meta["accuracy"]}

        if model_key == "naive":
            self._format(base, self.naive.priors, topk)
            base["note"] = "Probabilities are the training class priors; the prediction ignores the image."
            base["feedback"] = {"level": "low_confidence",
                                "message": "The majority-class baseline ignores the image and always "
                                           "predicts the most common training class."}
            return base

        if model_key == "classical":
            self._format(base, self._classical_proba(image), topk)
            base["note"] = "Approximate probabilities (softmax over standardized SVM margins)."
            base["feedback"] = {"level": "low_confidence",
                                "message": "Classical HOG + SVM is a weak baseline (4.4% test accuracy); "
                                           "treat the ranking loosely."}
            return base

        self._format(base, self.deep.predict_proba([image])[0], topk)
        conf = base["confidence"]
        if conf < CONFIDENCE_THRESHOLD:
            base["feedback"] = {"level": "low_confidence",
                                "message": f"Only {conf:.0%} confident. The photo may be unclear, cropped, "
                                           f"or an angle the model has not seen.",
                                "suggestion": "Try a clearer, well-lit photo of the whole car (3/4 front view)."}
        else:
            base["feedback"] = {"level": "confident",
                                "message": f"{conf:.0%} confident this is a {base['prediction']}."}
        return base


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
