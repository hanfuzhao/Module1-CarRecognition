# Trained models

This directory holds the **trained model artifacts** (weights), following the
project structure where `models/` is the directory for trained models and the
**model code** lives in [`../scripts/model.py`](../scripts/model.py).

| File | Model | Code |
|------|-------|------|
| `naive_majority.pkl` | Naive baseline (majority class) | `scripts/model.py` → `NaiveBaseline` |
| `classical_hog_svm.pkl` | Classical: HOG features + linear SVM | `scripts/model.py` → `ClassicalModel` |
| `deep_resnet50_mlp.pt` | Deep (deployed): frozen ResNet50 features + MLP head | `scripts/model.py` → `DeepModel` |

The deep model stores only the trained MLP head (~4.6 MB); the frozen ResNet50
backbone is fetched from torchvision at runtime. Regenerate everything with
`python setup.py`.
