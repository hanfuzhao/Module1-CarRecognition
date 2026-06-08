# Attribution: authored with AI assistance (Anthropic Claude, https://claude.ai).
# Data source: Stanford Cars via tanganke/stanford_cars on the Hugging Face Hub
#   https://huggingface.co/datasets/tanganke/stanford_cars
"""
Fetch the Stanford Cars dataset and write project metadata.

Data is pulled from the `tanganke/stanford_cars` mirror on the Hugging Face
Hub (see scripts/data.py for details) and cached by the `datasets` library.
This script does not duplicate the ~16k images onto disk; it only materializes
`data/raw/stanford-cars/metadata.json` (class names + counts), which the app
and EDA consume. The training pipeline reads images straight from the cache.

Run:  python scripts/make_dataset.py
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from scripts import data


def main():
    print("Loading Stanford Cars splits from Hugging Face (cached after first run)...")
    train = data.load_split("train")
    test = data.load_split("test")
    class_names = data.get_class_names(train)

    data.write_metadata(class_names, train.num_rows, test.num_rows)

    print(f"  train images : {train.num_rows}")
    print(f"  test images  : {test.num_rows}")
    print(f"  classes      : {len(class_names)}")
    print(f"  corruption splits available: {', '.join(data.CORRUPTION_SPLITS)}")
    print(f"\nWrote {data.METADATA_PATH}")


if __name__ == "__main__":
    main()
