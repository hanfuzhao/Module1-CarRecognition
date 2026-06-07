# Module 1 Project — Computer Vision

> 540 Summer · Module 1 · Project 1
> **Due:** Wed Jun 10, 2026 11:59pm · 150 points

## Project Topic
*(TBD — choose a CV domain. Must be NEW work, not reused from another course/research/job.)*

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

## Setup
```bash
pip install -r requirements.txt
python setup.py        # get data, build features, train model
python main.py         # run UI / project
```
