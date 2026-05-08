"""
Calculate ViT sequence lengths for all paragraph crop images in the dataset.
"""

import json
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

DATASET_JSON = "theseus_ocr_dataset/theseus_ocr_dataset.json"
CROPS_ROOT   = "theseus_ocr_dataset"


def calculate_seq_len(height, width, patch_h=16, patch_w=16, use_cls=True):
    n_h = (height + patch_h - 1) // patch_h
    n_w = (width + patch_w - 1) // patch_w
    seq_len = n_h * n_w
    if use_cls:
        seq_len += 1
    return seq_len


def main() -> None:
    with open(DATASET_JSON, encoding="utf-8") as f:
        records = json.load(f)

    print(f"Loaded {len(records)} records from '{DATASET_JSON}'")

    dims: list[tuple[int, int]] = []
    missing = 0
    for rec in tqdm(records, desc="Reading image sizes"):
        path = os.path.join(CROPS_ROOT, rec["image"])
        try:
            with Image.open(path) as img:
                w, h = img.size  # PIL gives (width, height)
                dims.append((h, w))
        except Exception:
            missing += 1

    if missing:
        print(f"⚠️  Skipped {missing} missing/unreadable images")

    seq_lengths = np.array([calculate_seq_len(h, w) for h, w in dims])
    heights     = np.array([h for h, w in dims])
    widths      = np.array([w for h, w in dims])

    print(f"\n── Image dimensions ──────────────────────")
    print(f"  Height  min={heights.min()}  max={heights.max()}  mean={heights.mean():.1f}  median={np.median(heights):.1f}")
    print(f"  Width   min={widths.min()}  max={widths.max()}  mean={widths.mean():.1f}  median={np.median(widths):.1f}")

    print(f"\n── Sequence lengths (patch 16×16, cls=True) ──")
    print(f"  min    : {seq_lengths.min()}")
    print(f"  max    : {seq_lengths.max()}")
    print(f"  mean   : {seq_lengths.mean():.1f}")
    print(f"  median : {np.median(seq_lengths):.1f}")
    p95   = int(np.percentile(seq_lengths, 95))
    p99   = int(np.percentile(seq_lengths, 99))
    p995  = int(np.percentile(seq_lengths, 99.5))
    p999  = int(np.percentile(seq_lengths, 99.9))
    p9999 = int(np.percentile(seq_lengths, 99.99))
    print(f"  p95    : {p95}  ← recommended max_len")
    print(f"  p99    : {p99}")
    print(f"  p99.5  : {p995}")
    print(f"  p99.9  : {p999}")
    print(f"  p99.99 : {p9999}")

    print()
    for pct, val in [(95, p95), (99, p99), (99.5, p995), (99.9, p999), (99.99, p9999)]:
        covered = int((seq_lengths <= val).sum())
        print(f"  Sequences ≤ p{pct:<5} ({val:>5}): {covered:>6}/{len(seq_lengths)} ({covered/len(seq_lengths)*100:.2f}%) covered")

    print(f"\n── Distribution ──────────────────────────")
    buckets = [1, 50, 100, 200, 500, 1000, 2000, int(seq_lengths.max()) + 1]
    for lo, hi in zip(buckets, buckets[1:]):
        count = int(((seq_lengths >= lo) & (seq_lengths < hi)).sum())
        pct = count / len(seq_lengths) * 100
        print(f"  [{lo:>5}, {hi:>5})  {count:>6} records  ({pct:.1f}%)")


if __name__ == "__main__":
    main()
