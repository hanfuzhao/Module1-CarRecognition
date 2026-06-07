# 🚗 Car Type Recognition - Module 1 Project

> **540 Summer · Module 1 · Project 1**  
> **Due:** Wed Jun 10, 2026 11:59pm · 150 points  
> **GitHub:** https://github.com/hanfuzhao/Module1-CarRecognition

## 🚀 Deploy to Render (One-Click)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/hanfuzhao/Module1-CarRecognition)

**Or follow:** [DEPLOYMENT.md](DEPLOYMENT.md) for manual deployment (5 minutes)

---

## Project Topic
**Car Type Recognition (AI识车)** — Fine-grained image classification on Stanford Cars dataset (196 classes, ~16k images). Maps to DZD's vehicle recognition feature while demonstrating long-tail robustness and confidence-based filtering mechanisms.

## Requirements Checklist

### Required Modeling Approaches (all three must be in repo)
- [ ] **Naive baseline** — e.g. mean predictor / majority class / random
- [ ] **Classical ML model** — e.g. logistic regression, random forest, gradient boosting, SVM
- [ ] **Deep learning (neural net)** model
- [ ] One of the above selected as the final **deployed** model
- [ ] Docs clearly stating where each model lives

### Required Experimentation (≥1 focused experiment)
- [ ] Well-motivated, clearly described, properly interpreted
- Options: sensitivity analysis · robustness/noise/dist-shift · preprocessing comparison · ablation

### Interactive Application
- [ ] Publicly accessible on the internet
- [ ] Live ≥ 1 week after submission
- [ ] **Inference only** (no training in the app)
- [ ] Strong UX — "investor demo" quality (basic Streamlit not enough)
- ⚠️ If app doesn't run when graded → **0** for that portion

### Novelty / Approach
- [ ] Either a novel dataset/problem, OR an existing problem with a clearly explained novel approach
- [ ] If existing: reference prior approaches + aim for SOTA-ish results OR better insight/explainability

## Deliverables

### 1. Written Report (NeurIPS/ICML paper · white paper · or technical report)
- [ ] Problem Statement
- [ ] Data Sources
- [ ] Related Work (literature review)
- [ ] Evaluation Strategy & Metrics (metric justification — *critical*)
- [ ] Modeling Approach
  - [ ] Data Processing Pipeline (rationale per step)
  - [ ] Hyperparameter Tuning Strategy
  - [ ] Models Evaluated (naive / classical / DL + rationale)
- [ ] Results (quantitative comparison, visualizations, confusion matrices)
- [ ] Error Analysis (5 specific mispredictions, root causes, mitigations)
- [ ] Experiment Write-Up (plan, results, interpretation, recommendations)
- [ ] Conclusions
- [ ] Future Work ("another semester?")
- [ ] Commercial Viability Statement
- [ ] Ethics Statement

### 2. In-class Pitch (5 min hard stop)
- [ ] Problem & Motivation · Approach Overview · Live Demo · Results/Insights

### 3. Code & Deployment
- [ ] Public/private GitHub repo with full codebase
- [ ] Live deployed web/mobile app link
- [ ] Git best practices: branches + PRs + PR reviews before merge to main

## Code Quality Rules
- Notebooks only in `notebooks/` (not graded — exploration only)
- All code modularized into classes/functions
- No loose executable code outside functions or `if __name__ == "__main__"`
- Descriptive names, docstrings, comments
- Attribute any external/AI code with a source link at top of file

## Implementation Overview

### Three Modeling Approaches

1. **Naive Baseline**: Majority class classifier + random classifier
   - Location: `scripts/model.py::NaiveBaseline`

2. **Classical ML**: HOG features + SVM / Random Forest  
   - Location: `scripts/model.py::ClassicalMLModel`
   - Feature pipeline: color histograms + HOG + PCA

3. **Deep Learning**: Fine-tuned ResNet50 / EfficientNet + CLIP zero-shot
   - Location: `scripts/model.py::DeepLearningModel`
   - Transfer learning from ImageNet pretrained weights

### Experiment: Long-Tail Performance Analysis & Confidence Filtering
- Measure accuracy gap between head classes (frequent) vs tail classes (rare)
- Implement confidence-based rejection mechanism
- Evaluate precision-recall curves at different confidence thresholds
- Location: `scripts/experiment.py`

### Key Novelty
CLIP zero-shot vs fine-tuned performance comparison + rejection-based UX improvement (when uncertain, ask user to retake photo)

## Deliverables Summary

### 1. Technical Report ✓
**See:** `TECHNICAL_REPORT.md` (13 sections, full NeurIPS-style report)
- Problem Statement · Data Sources · Related Work
- Evaluation Metrics · Modeling Approach (data pipeline, hyperparameter tuning)
- Results (82% top-1 accuracy) · Error Analysis (5 mispredictions)
- **Novel Experiment:** Long-tail robustness + confidence-based rejection
- Commercial Viability & Ethics Statement

### 2. In-class Pitch (5 min)
**Topics to Cover:**
- Problem: Long-tail car recognition (DZD context)
- Approach: 3-model comparison (Naive → Classical → ResNet50)
- Live Demo: Run main.py, upload car photo, show predictions + confidence feedback
- Results: 82% accuracy with confidence filtering; 16% head/tail gap

### 3. Code & Deployment
- **GitHub:** All code organized by feature (see PR structure below)
- **Live App:** Flask web app with modern UI (see "Run Interactive App" section)

## PR & Git Best Practices

This project demonstrates complete git workflow with **6 PRs**:

1. **PR #1: Dataset Setup & EDA** — Download Stanford Cars, exploratory analysis
2. **PR #2: Baseline Models** — Naive (majority/random) + Classical ML (HOG+SVM)
3. **PR #3: Deep Learning** — Fine-tuned ResNet50 + CLIP zero-shot
4. **PR #4: Experiment Pipeline** — Long-tail analysis + confidence filtering
5. **PR #5: Interactive Flask App** — Web UI with modern design
6. **PR #6: Final Report & Docs** — TECHNICAL_REPORT.md + this README

**View commit history:**
```bash
git log --oneline develop     # See all merged PRs
git show <commit-hash>        # Examine each feature
```

## Setup & Quick Start

### Prerequisites
```bash
python 3.9+
pip install -r requirements.txt
```

### Full Setup (from scratch)
```bash
# 1. Download dataset (~1.5 GB)
python scripts/make_dataset.py

# 2. Run exploratory analysis
python scripts/eda.py
# Outputs: data/outputs/sample_images.png, class_distribution.png

# 3. Train all models (baseline + DL)
python setup.py
# Outputs: models/naive_majority.pkl, classical_svm.pkl, resnet50_best.pt

# 4. Run experiments (long-tail + confidence)
python scripts/experiment.py
# Outputs: data/outputs/longtail_analysis.json, confidence_curve.png
```

### Run Interactive App (Production)
```bash
python main.py
# Server running on http://localhost:5000
# Open in browser → Upload car photo → Get prediction + guidance
```

## Project Structure
```
Module1_Project1/
├── README.md                    ← You are here
├── TECHNICAL_REPORT.md          ← Full research paper (read this!)
├── requirements.txt
├── setup.py                     ← Training entrypoint
├── main.py                      ← Flask web app
│
├── scripts/
│   ├── model.py                 ← NaiveBaseline, ClassicalMLModel, DeepLearningModel
│   ├── build_features.py        ← PyTorch DataLoaders, preprocessing
│   ├── train_dl.py              ← ResNet50 training loop, CLIP evaluation
│   ├── eda.py                   ← Class distribution, long-tail analysis
│   └── experiment.py            ← LongTailAnalyzer, ConfidenceFilterExperiment
│
├── templates/
│   └── index.html               ← Modern drag-drop UI
│
├── static/
│   ├── css/style.css            ← Gradient design, animations
│   └── js/app.js                ← Real-time image preview, API calls
│
├── data/
│   ├── raw/stanford-cars/       ← Original dataset (train/test by class)
│   ├── processed/               ← (Optional) Extracted features
│   └── outputs/                 ← Analysis results, visualizations
│
└── models/
    ├── naive_majority.pkl       ← Baseline models
    ├── classical_svm.pkl
    └── resnet50_best.pt         ← Best validation checkpoint

notebooks/                       ← Exploration notebooks (not graded)
```

## Key Models & Locations

| Model | File | Accuracy | Use Case |
|-------|------|----------|----------|
| Naive (Majority) | `scripts/model.py::NaiveBaseline` | 1% | Performance floor |
| Classical (HOG+SVM) | `scripts/model.py::ClassicalMLModel` | 48% | Interpretable baseline |
| **Deep Learning (ResNet50)** | `scripts/model.py::DeepLearningModel` | **82%** | **Production choice** |
| CLIP Zero-Shot | `scripts/train_dl.py::CLIPZeroShot` | 68% | No fine-tuning needed |

## Key Metrics

| Metric | Value | Context |
|--------|-------|---------|
| **Top-1 Accuracy** | 82% | ResNet50 on test set |
| Head Classes Accuracy | 88% | Frequent car types (>50 samples) |
| Tail Classes Accuracy | 72% | Rare types (<50 samples) |
| **Performance Gap** | 16% | Head vs tail (long-tail robustness) |
| Coverage @ 60% confidence | 72% | Accuracy 87% when accepting most queries |
| Coverage @ 70% confidence | 48% | Accuracy 91% (high-precision mode) |

## Experiment: Long-Tail + Confidence Filtering

**Problem:** Rare car types are hard to recognize (72% vs 88% on frequent types)

**Solution:** Confidence-based rejection (UX improvement)
- Model rejects low-confidence predictions
- Instead of wrong answer, ask user: "I'm not sure. Try a different angle or lighting."
- Trade-off: Accuracy improves but coverage decreases

**Results:** See `data/outputs/confidence_curve.png` and `TECHNICAL_REPORT.md` Section 9.2

## Deployment Notes

### For Grading (Recommended)
```bash
# Quick validation that everything works
python main.py
# Access http://localhost:5000
# Upload sample car image → should get prediction
```

### For Production Deployment
1. Pre-download dataset: `python scripts/make_dataset.py`
2. Pre-train models: `python setup.py` (or use provided `models/*.pkl`)
3. Deploy Flask app on cloud (AWS, GCP, Azure)
4. Optimize: Model quantization, batch inference, caching

### Expected Performance
- Inference time: ~50ms (GPU) / ~200ms (CPU) per image
- Model size: ~100MB (ResNet50)
- Memory: ~1GB RAM per worker

## Citation
```bibtex
@misc{module1_project,
  author = {Leo},
  title = {Fine-Grained Car Type Recognition: Long-Tail Robustness and Confidence-Based UX},
  year = {2026},
  school = {Class 540}
}
```

**Data Source:** Krause, J., Stark, M., Deng, J., & Fei-Fei, L. (2013). 
3D Object Representations for Fine-Grained Categorization. ICCV.
