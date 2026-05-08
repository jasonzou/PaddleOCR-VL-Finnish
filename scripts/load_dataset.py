"""
Load FIN-13K as a HuggingFace Dataset.

Usage:
    from load_dataset import load_fin13k
    dataset = load_fin13k()

Or:
    python load_dataset.py
"""

from datasets import Dataset, Features, Value, Image
import os
import argparse
from pathlib import Path


def load_fin13k(data_dir: str = None):
    """Load FIN-13K dataset as a HuggingFace Dataset."""
    if data_dir is None:
        data_dir = Path(__file__).parent
    else:
        data_dir = Path(data_dir)

    if data_dir.name == "nlf_ocr":
        parquet_file = str(data_dir / "train-*-of-*.parquet")
    else:
        train_parquet = data_dir / "train.parquet"
        if train_parquet.exists():
            parquet_file = str(train_parquet)
        else:
            parquet_file = data_dir / "data_std.parquet"
            if not parquet_file.exists():
                parquet_file = data_dir / "data.parquet"

    features = Features({
        "file_name": Value("string"),
        "image": Image(),
        "text": Value("string"),
    })

    dataset = Dataset.from_parquet(str(parquet_file), features=features, base_path=data_dir)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="swe-11k", help="Dataset name (swe-11k, digi-data, natlib-data, nlf_ocr)")
    parser.add_argument("--sample", type=int, default=0, help="Randomly sample N examples and export images")
    args = parser.parse_args()

    data_dir = Path(__file__).parent / args.dataset
    print(f"Loading {args.dataset} dataset...")
    dataset = load_fin13k(str(data_dir))
    print(f"Dataset loaded successfully!")
    print(f"Number of examples: {len(dataset)}")
    print(f"Features: {dataset.features}")

    if args.sample > 0:
        import random
        random.seed(42)
        indices = random.sample(range(len(dataset)), k=args.sample)
        out_dir = data_dir / "sample_images"
        out_dir.mkdir(exist_ok=True)
        for i in indices:
            ex = dataset[i]
            img = ex["image"]
            out_path = out_dir / ex["file_name"]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path)
            print(f"{ex['file_name']}: {ex['text']}")
        print(f"\nExported {args.sample} images to {out_dir}")
    else:
        print("\nFirst example:")
        print(dataset[0])
