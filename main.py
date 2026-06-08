"""
Interactive web app for car-type recognition (inference only).

Upload a car photo -> the deployed deep model (frozen ResNet50 features + MLP
head) returns the top-5 predicted makes/models with confidence, plus
confidence-aware guidance (low confidence -> suggest a clearer photo).

Run:  python main.py    then open http://localhost:5000
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify

from scripts.model import DeepModel

MODEL_PATH = Path("models/deep_resnet50_mlp.pt")
METADATA_PATH = Path("data/raw/stanford-cars/metadata.json")
CONFIDENCE_THRESHOLD = 0.40  # below this we ask the user for a clearer photo


class PredictionService:
    """Loads the deployed deep model once and serves predictions."""

    def __init__(self):
        self.model = DeepModel.load(str(MODEL_PATH))
        self.model._ensure_backbone()  # warm up the frozen ResNet50 backbone
        self.class_names = self.model.class_names or self._fallback_names()

    def _fallback_names(self):
        if METADATA_PATH.exists():
            return json.loads(METADATA_PATH.read_text())["class_names"]
        return [f"Class {i}" for i in range(self.model.num_classes)]

    def predict(self, image: Image.Image, topk: int = 5) -> dict:
        proba = self.model.predict_proba([image])[0]
        order = np.argsort(proba)[::-1][:topk]
        top = [{"label": self.class_names[i], "confidence": float(proba[i])} for i in order]
        confidence = top[0]["confidence"]

        if confidence < CONFIDENCE_THRESHOLD:
            feedback = {
                "level": "low_confidence",
                "message": f"Only {confidence:.0%} confident. The photo may be unclear, "
                           f"cropped, or an angle the model hasn't seen.",
                "suggestion": "📸 Try a clearer, well-lit photo of the whole car (3/4 front view).",
            }
        else:
            feedback = {
                "level": "confident",
                "message": f"{confidence:.0%} confident this is a {top[0]['label']}.",
            }

        return {"prediction": top[0]["label"], "confidence": confidence,
                "top_k": top, "feedback": feedback}


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
    ok = MODEL_PATH.exists()
    return jsonify({"status": "ok" if ok else "model_missing",
                    "num_classes": get_service().model.num_classes if ok else 0})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files or request.files["image"].filename == "":
        return jsonify({"error": "No image provided"}), 400
    try:
        raw = request.files["image"].read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return jsonify({"error": "Could not read image file"}), 400

    try:
        return jsonify(get_service().predict(image))
    except Exception as e:  # pragma: no cover - defensive
        return jsonify({"error": f"Inference failed: {e}"}), 500


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "File too large (max 16MB)"}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("Loading model and warming up backbone...")
    get_service()
    print(f"Ready. Open http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
