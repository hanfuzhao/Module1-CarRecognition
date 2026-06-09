# Built with AI assistance (Claude). Uses torch/torchvision, scikit-learn, matplotlib.
"""
End-to-end pipeline: build data artifacts, train all three models, evaluate on
the test set, run the robustness and confidence experiments, and write metrics
and plots to data/outputs and trained models to models/.

Run: python setup.py

ResNet features are cached under data/processed so a re-run is fast and a
failure midway does not lose completed work.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts import data
from scripts.model import (
    NaiveBaseline,
    ClassicalModel,
    DeepModel,
    evaluate_predictions,
    topk_accuracy,
)
from scripts import experiment as exp
from scripts.build_features import deep_features as cached_deep_features
from scripts.build_features import hog_features as cached_hog_features

PROCESSED = Path("data/processed")
OUTPUTS = Path("data/outputs")
MODELS = Path("models")


def ensure_dirs():
    """Create output directories if missing (called from main)."""
    for d in (PROCESSED, OUTPUTS, MODELS):
        d.mkdir(parents=True, exist_ok=True)


NUM_CLASSES = 196
ROBUSTNESS_CORRUPTIONS = ["gaussian_noise", "motion_blur", "jpeg_compression", "pixelate"]
ROBUSTNESS_SAMPLE = 2500  # images per corruption split (kept fast; noted in report)


def plot_model_comparison(results: dict):
    names = ["Naive (majority)", "Naive (random)", "Classical (HOG+SVM)", "Deep (ResNet50+MLP)"]
    accs = [
        results["naive_majority"]["accuracy"],
        results["naive_random"]["accuracy"],
        results["classical"]["accuracy"],
        results["deep"]["accuracy"],
    ]
    plt.figure(figsize=(7, 4))
    bars = plt.bar(names, [a * 100 for a in accs], color=["#bbb", "#bbb", "#6a9", "#36c"])
    plt.ylabel("Top-1 accuracy (%)")
    plt.title("Model comparison on Stanford Cars test set")
    for b, a in zip(bars, accs):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{a*100:.1f}%", ha="center")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "model_comparison.png", dpi=120)
    plt.close()


def plot_confidence_curve(rows):
    thr = [r["threshold"] for r in rows]
    acc = [r["accuracy"] * 100 for r in rows]
    cov = [r["coverage"] * 100 for r in rows]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(thr, acc, "o-", color="#36c", label="Accuracy on accepted")
    ax1.set_xlabel("Confidence threshold")
    ax1.set_ylabel("Accuracy (%)", color="#36c")
    ax2 = ax1.twinx()
    ax2.plot(thr, cov, "s--", color="#e67", label="Coverage")
    ax2.set_ylabel("Coverage (%)", color="#e67")
    plt.title("Selective prediction: accuracy vs coverage")
    fig.tight_layout()
    plt.savefig(OUTPUTS / "confidence_curve.png", dpi=120)
    plt.close()


def plot_robustness(clean_acc, corruption_accs):
    names = ["clean"] + list(corruption_accs.keys())
    accs = [clean_acc] + list(corruption_accs.values())
    plt.figure(figsize=(7, 4))
    colors = ["#36c"] + ["#e67"] * len(corruption_accs)
    bars = plt.bar(names, [a * 100 for a in accs], color=colors)
    plt.ylabel("Top-1 accuracy (%)")
    plt.title("Robustness to input corruptions (deep model)")
    for b, a in zip(bars, accs):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{a*100:.1f}%", ha="center")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "robustness.png", dpi=120)
    plt.close()


def plot_confusion(y_true, y_pred):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    np.save(OUTPUTS / "confusion_matrix.npy", cm)
    plt.figure(figsize=(6, 5))
    plt.imshow(np.log1p(cm), cmap="viridis")
    plt.colorbar(label="log(1 + count)")
    plt.title("Confusion matrix (196 classes)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "confusion_matrix.png", dpi=120)
    plt.close()
    return cm


def save_sample_images(class_names):
    ds = data.load_split("train")
    rows = ds[0:8]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, im, lbl in zip(axes.ravel(), rows["image"], rows["label"]):
        ax.imshow(im)
        ax.set_title(class_names[lbl], fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "sample_images.png", dpi=110)
    plt.close()


def error_analysis(y_true, proba, class_names, k=5):
    y_pred = proba.argmax(1)
    conf = proba.max(1)
    wrong = np.where(y_pred != y_true)[0]
    # pick confident mistakes (most informative)
    wrong_sorted = wrong[np.argsort(-conf[wrong])]
    picks = wrong_sorted[:k]
    cases = []
    for i in picks:
        cases.append(
            {
                "test_index": int(i),
                "true": class_names[y_true[i]],
                "predicted": class_names[y_pred[i]],
                "confidence": float(conf[i]),
            }
        )
    return cases


def main():
    t_start = time.time()
    ensure_dirs()
    results = {}

    print("\n[0/6] Metadata + EDA samples")
    train_ds = data.load_split("train")
    class_names = data.get_class_names(train_ds)
    n_test = data.load_split("test").num_rows
    data.write_metadata(class_names, train_ds.num_rows, n_test)
    save_sample_images(class_names)
    train_labels = data.all_labels(train_ds)
    train_counts = np.bincount(train_labels, minlength=NUM_CLASSES)

    print("\n[1/6] Deep features (frozen ResNet50)")
    deep = DeepModel(num_classes=NUM_CLASSES, class_names=class_names)
    Xtr, ytr = cached_deep_features(deep, "train")
    Xte, yte = cached_deep_features(deep, "test")

    print("\n[2/6] Naive baselines")
    nb = NaiveBaseline("majority").fit(ytr)
    nr = NaiveBaseline("random").fit(ytr)
    results["naive_majority"] = evaluate_predictions(yte, nb.predict(len(yte)))
    results["naive_random"] = evaluate_predictions(yte, nr.predict(len(yte)))
    nb.save(MODELS / "naive_majority.pkl")
    print(f"  majority acc={results['naive_majority']['accuracy']:.4f}")
    print(f"  random   acc={results['naive_random']['accuracy']:.4f}")

    print("\n[3/6] Classical model (HOG + LinearSVM)")
    clf = ClassicalModel()
    Htr, _ = cached_hog_features(clf, "train")
    Hte, _ = cached_hog_features(clf, "test")
    clf_path = MODELS / "classical_hog_svm.pkl"
    if clf_path.exists():
        import pickle
        print("  [cache] classical model")
        with open(clf_path, "rb") as f:
            clf = pickle.load(f)
    else:
        t0 = time.time()
        clf.fit(Htr, ytr)
        print(f"  fit done in {time.time() - t0:.0f}s")
        clf.save(clf_path)
    classical_pred = clf.predict(Hte)
    results["classical"] = evaluate_predictions(yte, classical_pred)
    print(f"  classical acc={results['classical']['accuracy']:.4f}")

    print("\n[4/6] Deep model (MLP head on ResNet50 features)")
    deep.fit_features(Xtr, ytr, epochs=60)
    proba_te = deep.predict_proba_features(Xte)
    deep_pred = proba_te.argmax(1)
    results["deep"] = {
        "accuracy": float((deep_pred == yte).mean()),
        "top5_accuracy": topk_accuracy(yte, proba_te, k=5, num_classes=NUM_CLASSES),
    }
    deep.save(MODELS / "deep_resnet50_mlp.pt")
    print(f"  deep acc={results['deep']['accuracy']:.4f}  top5={results['deep']['top5_accuracy']:.4f}")

    print("\n[5/6] Experiments")
    # confidence / selective prediction
    conf_rows = exp.confidence_rejection(yte, proba_te)
    with open(OUTPUTS / "confidence_analysis.json", "w") as f:
        json.dump(conf_rows, f, indent=2)
    plot_confidence_curve(conf_rows)

    # head/tail + per-class
    pca = exp.per_class_accuracy(yte, deep_pred, NUM_CLASSES)
    ht = exp.head_tail_gap(yte, deep_pred, train_counts, NUM_CLASSES)
    results["head_tail"] = ht

    # robustness on corruption splits (deep model)
    corruption_accs = {}
    for corr in ROBUSTNESS_CORRUPTIONS:
        Xc, yc = cached_deep_features(deep, corr, limit=ROBUSTNESS_SAMPLE)
        pc = deep.predict_proba_features(Xc).argmax(1)
        corruption_accs[corr] = float((pc == yc).mean())
        print(f"  robustness[{corr}] acc={corruption_accs[corr]:.4f}")
    results["robustness"] = exp.summarize_robustness(results["deep"]["accuracy"], corruption_accs)
    plot_robustness(results["deep"]["accuracy"], corruption_accs)

    print("\n[6/6] Plots, confusion matrix, error analysis")
    plot_model_comparison(results)
    cm = plot_confusion(yte, deep_pred)
    # top confusion pairs
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)
    pairs = []
    for _ in range(10):
        i, j = np.unravel_index(cm_off.argmax(), cm_off.shape)
        if cm_off[i, j] == 0:
            break
        pairs.append({"true": class_names[i], "predicted": class_names[j], "count": int(cm_off[i, j])})
        cm_off[i, j] = 0
    results["top_confusions"] = pairs
    results["error_cases"] = error_analysis(yte, proba_te, class_names, k=5)

    results["meta"] = {
        "dataset": data.HF_DATASET,
        "num_classes": NUM_CLASSES,
        "num_train": int(train_ds.num_rows),
        "num_test": int(len(yte)),
        "device": deep.device,
        "runtime_sec": round(time.time() - t_start, 1),
    }

    with open(OUTPUTS / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n==================== SUMMARY ====================")
    print(f"Naive (majority)   : {results['naive_majority']['accuracy']*100:5.2f}%")
    print(f"Naive (random)     : {results['naive_random']['accuracy']*100:5.2f}%")
    print(f"Classical (HOG+SVM): {results['classical']['accuracy']*100:5.2f}%")
    print(f"Deep (ResNet50+MLP): {results['deep']['accuracy']*100:5.2f}%  (top5 {results['deep']['top5_accuracy']*100:.2f}%)")
    print(f"Head/tail gap      : {ht['gap']*100:.2f}%  (head {ht['head_accuracy']*100:.1f} / tail {ht['tail_accuracy']*100:.1f})")
    print(f"Mean corruption acc: {results['robustness']['mean_corruption_accuracy']*100:.2f}%")
    print(f"Runtime            : {results['meta']['runtime_sec']}s")
    print("Saved -> models/, data/outputs/metrics.json")


if __name__ == "__main__":
    main()
