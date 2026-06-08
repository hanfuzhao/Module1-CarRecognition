# Car-Type Recognition — Module 1 Project

> **540 Summer · Module 1 · Project 1** — fine-grained car make/model/year
> classification on Stanford Cars (196 classes).
>
> **Live demo:** https://HanfuZhao781-car-recognition.hf.space
> **GitHub:** https://github.com/hanfuzhao/Module1-CarRecognition
> **Report:** [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) · **Pitch:** [PITCH.md](PITCH.md) · **Deploy:** [DEPLOYMENT.md](DEPLOYMENT.md)

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
python scripts/make_dataset.py   # fetch dataset (streams from HF cache) + write metadata
python scripts/eda.py            # class-distribution stats + sample montage
python setup.py                  # train all 3 models + run experiments -> data/outputs/
python main.py                   # serve the web app at http://localhost:5000
```
First run downloads the dataset (~4.5 GB, cached). Training is ~3.5 min on
Apple-Silicon MPS afterwards (deep + HOG features are cached to
`data/processed/`).

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
│   ├── raw/stanford-cars/      # metadata.json (images stream from HF cache)
│   ├── processed/              # cached features (gitignored)
│   └── outputs/                # metrics.json + plots
└── notebooks/                  # exploration (not graded)
```

## Git workflow
Built across feature branches, each merged via a reviewed Pull Request into
`main` (see the repo's Pull Requests tab).

## Data & citation
Stanford Cars via the `tanganke/stanford_cars` Hugging Face mirror (canonical
train/test + 7 corruption splits). Krause et al., *3D Object Representations for
Fine-Grained Categorization*, ICCV 2013.
