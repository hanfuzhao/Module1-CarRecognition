"""
Setup script: Download data, train baseline models, evaluate.
Run: python setup.py
"""

import sys
import json
from pathlib import Path
from scripts.model import NaiveBaseline, ClassicalMLModel
from scripts.build_features import get_image_paths


def train_baseline_models(data_dir: str = "data/raw/stanford-cars", output_dir: str = "models"):
    """Train naive baseline and classical ML models."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("BASELINE MODELS: Naive + Classical ML")
    print("=" * 60)

    # Get image paths and labels
    print("\n[1/3] Loading image paths and labels...")
    train_paths, train_labels = get_image_paths(data_dir, split="train")
    test_paths, test_labels = get_image_paths(data_dir, split="test")

    print(f"  Train: {len(train_paths)} images, {len(set(train_labels))} classes")
    print(f"  Test:  {len(test_paths)} images")

    # Train Naive Baseline (Majority Class)
    print("\n[2/3] Training Naive Baseline (Majority Class)...")
    naive_majority = NaiveBaseline(strategy="majority")
    naive_majority.train(train_paths, train_labels)
    results_majority = naive_majority.evaluate(test_paths, test_labels)
    print(f"  Accuracy: {results_majority['accuracy']:.4f}")

    # Train Naive Baseline (Random)
    print("      Training Naive Baseline (Random)...")
    naive_random = NaiveBaseline(strategy="random")
    naive_random.train(train_paths, train_labels)
    results_random = naive_random.evaluate(test_paths, test_labels)
    print(f"  Accuracy: {results_random['accuracy']:.4f}")

    # Train Classical ML (SVM + HOG)
    print("\n[3/3] Training Classical ML (HOG + SVM)...")
    classical_ml = ClassicalMLModel(model_type="svm")
    classical_ml.train(train_paths, train_labels)
    results_classical = classical_ml.evaluate(test_paths, test_labels)
    print(f"  Accuracy: {results_classical['accuracy']:.4f}")

    # Save models
    print("\n[✓] Saving models...")
    naive_majority.save(f"{output_dir}/naive_majority.pkl")
    naive_random.save(f"{output_dir}/naive_random.pkl")
    classical_ml.save(f"{output_dir}/classical_svm.pkl")

    # Summary
    summary = {
        "naive_majority_accuracy": float(results_majority["accuracy"]),
        "naive_random_accuracy": float(results_random["accuracy"]),
        "classical_ml_accuracy": float(results_classical["accuracy"]),
    }

    with open(f"{output_dir}/baseline_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nResults Summary:")
    print(f"  Naive (Majority): {results_majority['accuracy']:.4f}")
    print(f"  Naive (Random):   {results_random['accuracy']:.4f}")
    print(f"  Classical (SVM):  {results_classical['accuracy']:.4f}")

    return summary


if __name__ == "__main__":
    print("Setting up Stanford Cars Car Type Recognition project...")
    train_baseline_models()
    print("\n✓ Setup complete!")
