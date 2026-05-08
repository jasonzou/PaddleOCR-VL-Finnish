"""Smoke-test for the local HuggingFace dataset (both embed and no-embed modes)."""

import argparse
import os
from datasets import load_dataset, features

HF_DATASET_FOLDER = "theseus_ocr_hf"
NUM_SAMPLES = 6


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the local HF dataset.")
    parser.add_argument(
        "--folder", "-f",
        default=HF_DATASET_FOLDER,
        help="Dataset folder to test (default: %(default)s)",
    )
    args = parser.parse_args()

    parquet_files = sorted(
        os.path.join(args.folder, f)
        for f in os.listdir(args.folder)
        if f.endswith(".parquet")
    )
    if not parquet_files:
        print(f"No parquet files found in '{args.folder}/'")
        return

    print(f"Loading dataset from '{args.folder}/' ({len(parquet_files)} shard(s))...")
    ds = load_dataset("parquet", data_files=parquet_files, split="train")

    # Detect mode by checking column names
    embed_mode = "image" in ds.column_names
    if embed_mode:
        ds = ds.cast_column("image", features.Image())

    print(f"  Rows     : {len(ds)}")
    print(f"  Columns  : {ds.column_names}")
    print(f"  Features : {ds.features}")
    print(f"  Mode     : {'embedded images' if embed_mode else 'separate image paths'}")

    print(f"\nSample records (first {NUM_SAMPLES}):")
    for i, row in enumerate(ds.select(range(min(NUM_SAMPLES, len(ds))))):
        print(f"\n  [{i}] pdf_file : {row['pdf_file']}")
        print(f"       page     : {row['page']}")
        print(f"       text     : {row['text'][:80]!r}")

        if embed_mode:
            img = row["image"]
            out_path = f"test_sample_{i}.jpg"
            img.convert("RGB").save(out_path, format="JPEG")
            print(f"       image    : {img.size} px, mode={img.mode} → {out_path}")
        else:
            print(f"       image_path: {row['image_path']}")


if __name__ == "__main__":
    main()
