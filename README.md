# Car-Type Recognition — Module 1 Project

> **540 Summer · Module 1 · Project 1** — fine-grained car make/model/year
> classification on Stanford Cars (196 classes).
>
> **Live demo:** https://HanfuZhao781-car-recognition.hf.space
> **GitHub:** https://github.com/hanfuzhao/Module1-CarRecognition
> **Report:** [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) · **Pitch:** [PITCH.md](PITCH.md)

All metrics below are produced by `python setup.py` on the real Stanford Cars
test set and saved to `data/outputs/metrics.json` — fully reproducible.

## Results (real, test set = 8,041 images)

| Model | Top-1 | Top-5 | Location |
|---|---|---|---|
| Naive — majority class | 0.85% | – | `scripts/model.py::NaiveBaseline` |
| Naive — random | 0.60% | – | `scripts/model.py::NaiveBaseline` |
| Classical — HOG + linear SVM | 4.41% | – | `scripts/model.py::ClassicalModel` |
| **Deep — ResNet50 features + MLP (deployed)** | **52.16%** | **79.98%** | `scripts/model.py::DeepModel` |

**Experiment (corruption robustness):** robust to JPEG (+2pt) and motion blur
(+1pt), −5.6pt under noise, and **collapses under pixelation (−51pt)**.
**Confidence gating** lifts accuracy to **79% @ 43% coverage** and **90% @ 19%
coverage**. Full write-up: [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).

## Quick start
```bash
pip install -r requirements.txt
python scripts/make_dataset.py   # download the full ~16k images into data/raw/ + metadata
python scripts/eda.py            # class-distribution stats + sample montage
python setup.py                  # train all 3 models + run experiments -> data/outputs/
python main.py                   # serve the web app at http://localhost:5000
```
`make_dataset.py` writes all 16,185 images to `data/raw/stanford-cars/` (~2 GB,
git-ignored); a one-image-per-class sample is committed under `data/raw/sample/`
so the data is visible in the repo without the bulk. Training is ~3.5 min on
Apple-Silicon MPS (features are cached to `data/processed/`). To just try the
deployed app without retraining, the trained model is already in `models/` — run
`python main.py` directly.

## The three required models
1. **Naive baseline** — majority-class / random (`NaiveBaseline`).
2. **Classical ML** — HOG features → linear SVM, trained via SGD (`ClassicalModel`).
3. **Deep learning (deployed)** — frozen ImageNet ResNet50 as a feature
   extractor + a trained MLP head (`DeepModel`). The backbone is fetched by
   torchvision at runtime, so the committed model artifact is only ~4.6 MB.

## Interactive app
`main.py` is a Flask app: upload a car photo → top-5 predictions with
confidence bars + confidence-aware feedback (below 40% confidence it asks for a
clearer photo, mitigating the pixelation/blur failure mode). Inference only —
no training in the app.

## Repository layout
```
├── main.py                     # Flask inference app (deployed model)
├── setup.py                    # trains all 3 models + runs experiments
├── requirements.txt
├── TECHNICAL_REPORT.md         # full write-up (real results)
├── scripts/
│   ├── data.py                 # HF dataset access + metadata
│   ├── make_dataset.py         # fetch dataset, write metadata.json
│   ├── eda.py                  # class distribution + samples
│   ├── model.py                # NaiveBaseline / ClassicalModel / DeepModel
│   └── experiment.py           # confidence gating, head/tail, robustness helpers
├── templates/ , static/        # web UI
├── models/                     # trained models (naive, classical, deep head)
├── data/
│   ├── raw/
│   │   ├── stanford-cars/      # full 16k images written here by make_dataset.py (gitignored) + metadata.json
│   │   └── sample/             # one image per class, committed for visibility
│   ├── processed/              # cached features (gitignored)
│   └── outputs/                # metrics.json + plots
└── notebooks/                  # exploration (not graded)
```

## Deployment
The app is deployed as a Docker Space on Hugging Face (live URL above). It runs
inference only: the trained head ships in `models/`, the ResNet50 backbone is
fetched at build time, and a scheduled GitHub Action pings `/health` so the
Space stays awake. To run the container yourself:
```bash
docker build -t car-recognition .
docker run -p 7860:7860 car-recognition   # http://localhost:7860
```

## Git workflow
Built across feature branches, each merged via a reviewed Pull Request into
`main` (see the repo's Pull Requests tab).

## Data & citation
Stanford Cars via the `tanganke/stanford_cars` Hugging Face mirror (canonical
train/test + 7 corruption splits). Krause et al., *3D Object Representations for
Fine-Grained Categorization*, ICCV 2013.
