"""
Experiment: Long-tail performance analysis + confidence-based rejection.
Investigates robustness on rare car types (tail classes).
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, confusion_matrix
import torch


class LongTailAnalyzer:
    """Analyze model performance on head vs tail classes."""

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, class_counts: list = None):
        """
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            class_counts: Number of training samples per class
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.class_counts = class_counts or np.bincount(y_true)
        self.num_classes = len(self.class_counts)

    def compute_per_class_accuracy(self):
        """Compute accuracy per class."""
        per_class_acc = {}
        for class_id in range(self.num_classes):
            mask = self.y_true == class_id
            if mask.sum() == 0:
                continue
            acc = (self.y_pred[mask] == class_id).mean()
            per_class_acc[class_id] = float(acc)
        return per_class_acc

    def split_head_tail(self, percentile: float = 50.0):
        """Split classes into head and tail based on training distribution."""
        threshold = np.percentile(self.class_counts, percentile)

        head_classes = np.where(self.class_counts >= threshold)[0]
        tail_classes = np.where(self.class_counts < threshold)[0]

        head_mask = np.isin(self.y_true, head_classes)
        tail_mask = np.isin(self.y_true, tail_classes)

        head_acc = (self.y_pred[head_mask] == self.y_true[head_mask]).mean() if head_mask.sum() > 0 else 0
        tail_acc = (self.y_pred[tail_mask] == self.y_true[tail_mask]).mean() if tail_mask.sum() > 0 else 0

        return {
            "head_threshold": float(threshold),
            "num_head_classes": len(head_classes),
            "num_tail_classes": len(tail_classes),
            "head_accuracy": float(head_acc),
            "tail_accuracy": float(tail_acc),
            "performance_gap": float(head_acc - tail_acc),
        }

    def analyze(self):
        """Full long-tail analysis."""
        per_class_acc = self.compute_per_class_accuracy()
        head_tail_split = self.split_head_tail()

        results = {
            "overall_accuracy": float((self.y_pred == self.y_true).mean()),
            "per_class_accuracy_stats": {
                "mean": float(np.mean(list(per_class_acc.values()))),
                "std": float(np.std(list(per_class_acc.values()))),
                "min": float(min(per_class_acc.values())),
                "max": float(max(per_class_acc.values())),
            },
            "head_vs_tail": head_tail_split,
        }

        return results


class ConfidenceFilterExperiment:
    """Evaluate model with confidence-based rejection."""

    def __init__(self, y_true: np.ndarray, confidences: np.ndarray):
        """
        Args:
            y_true: Ground truth labels
            confidences: Model confidence scores (0-1) for predictions
        """
        self.y_true = y_true
        self.confidences = confidences

    def evaluate_at_thresholds(self, thresholds: list = None):
        """Evaluate accuracy at different confidence thresholds."""
        if thresholds is None:
            thresholds = np.linspace(0.0, 1.0, 11)

        results = []
        for threshold in thresholds:
            # Only evaluate on predictions above threshold
            valid_mask = self.confidences >= threshold
            if valid_mask.sum() == 0:
                results.append({"threshold": float(threshold), "accuracy": 0, "coverage": 0, "rejected": len(self.y_true)})
                continue

            accuracy = (self.y_true[valid_mask] == self.y_true[valid_mask]).mean() if valid_mask.sum() > 0 else 0
            coverage = valid_mask.sum() / len(self.y_true)
            rejected = (~valid_mask).sum()

            results.append(
                {
                    "threshold": float(threshold),
                    "accuracy": float(accuracy),
                    "coverage": float(coverage),
                    "rejected": int(rejected),
                }
            )

        return results

    def plot_confidence_curve(self, output_path: str = "data/outputs/confidence_curve.png"):
        """Plot accuracy vs confidence threshold."""
        thresholds = np.linspace(0.0, 1.0, 11)
        results = self.evaluate_at_thresholds(thresholds.tolist())

        thresholds = [r["threshold"] for r in results]
        accuracies = [r["accuracy"] for r in results]
        coverages = [r["coverage"] for r in results]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Accuracy curve
        axes[0].plot(thresholds, accuracies, "o-", linewidth=2, markersize=6)
        axes[0].set_xlabel("Confidence Threshold")
        axes[0].set_ylabel("Accuracy (on accepted samples)")
        axes[0].set_title("Accuracy vs Confidence Threshold")
        axes[0].grid(True, alpha=0.3)

        # Coverage curve
        axes[1].plot(thresholds, coverages, "s-", color="orange", linewidth=2, markersize=6)
        axes[1].set_xlabel("Confidence Threshold")
        axes[1].set_ylabel("Coverage (% samples accepted)")
        axes[1].set_title("Coverage vs Confidence Threshold")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        print(f"✓ Saved confidence curve to {output_path}")

        return results


def run_long_tail_experiment(
    y_true: np.ndarray, y_pred: np.ndarray, class_counts: list, output_dir: str = "data/outputs"
):
    """Run full long-tail analysis experiment."""

    print("\n" + "=" * 60)
    print("EXPERIMENT: Long-Tail Performance Analysis")
    print("=" * 60)

    analyzer = LongTailAnalyzer(y_true, y_pred, class_counts)
    results = analyzer.analyze()

    print("\nResults:")
    print(f"  Overall Accuracy: {results['overall_accuracy']:.4f}")
    print(f"  Head Classes Accuracy: {results['head_vs_tail']['head_accuracy']:.4f}")
    print(f"  Tail Classes Accuracy: {results['head_vs_tail']['tail_accuracy']:.4f}")
    print(f"  Performance Gap: {results['head_vs_tail']['performance_gap']:.4f}")

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/longtail_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_confidence_experiment(y_pred: np.ndarray, confidences: np.ndarray, output_dir: str = "data/outputs"):
    """Run confidence-based rejection experiment."""

    print("\n" + "=" * 60)
    print("EXPERIMENT: Confidence-Based Rejection")
    print("=" * 60)

    exp = ConfidenceFilterExperiment(y_pred, confidences)

    results = exp.evaluate_at_thresholds()
    print("\nConfidence Threshold Analysis:")
    print("Threshold | Accuracy | Coverage | Rejected")
    print("-" * 45)
    for r in results:
        print(f"{r['threshold']:8.2f}  | {r['accuracy']:8.4f} | {r['coverage']:8.4f} | {r['rejected']:8d}")

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/confidence_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    exp.plot_confidence_curve(f"{output_dir}/confidence_curve.png")

    return results


if __name__ == "__main__":
    print("Experiment module ready. Use in training scripts.")
