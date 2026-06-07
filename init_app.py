"""
Initialize app for deployment: Download dataset and models if needed.
Run before deploying to production.
"""

import os
from pathlib import Path


def init_deployment():
    """Initialize all required files for deployment."""
    print("Initializing Car Type Recognition app...")

    # Create required directories
    dirs = [
        "data/raw/stanford-cars",
        "data/processed",
        "data/outputs",
        "models",
        "static/uploads",
    ]

    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {d}/")

    # Check if dataset exists
    if not (Path("data/raw/stanford-cars") / "metadata.json").exists():
        print("\n⚠️  Stanford Cars dataset not found.")
        print("For local development, run: python scripts/make_dataset.py")
        print("For deployment, models will be loaded if available.\n")

    # Check if models exist
    model_files = [
        "models/naive_majority.pkl",
        "models/classical_svm.pkl",
        "models/resnet50_best.pt",
    ]

    existing_models = [f for f in model_files if Path(f).exists()]

    if existing_models:
        print(f"✓ Found {len(existing_models)} pre-trained models:")
        for m in existing_models:
            print(f"  - {m}")
    else:
        print("\n⚠️  No pre-trained models found.")
        print("For full functionality, run: python setup.py")
        print("App will still work with available models.\n")

    print("✓ Initialization complete!")


if __name__ == "__main__":
    init_deployment()
