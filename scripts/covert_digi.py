#!/usr/bin/env python3
"""Convert digi-data to HuggingFace dataset format."""

import pandas as pd
from pathlib import Path
from PIL import Image
import io


def convert():
    data_dir = Path(__file__).parent / "digi-data"
    df = pd.read_parquet(data_dir / "data.parquet")

    rows = []
    for _, row in df.iterrows():
        if pd.isna(row["text"]) or not str(row["text"]).strip():
            continue

        img_path = data_dir / row["image"]
        if not img_path.exists():
            continue

        img = Image.open(img_path)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_data = img_bytes.getvalue()

        rows.append({
            "file_name": row["image"],
            "image": img_data,
            "text": row["text"],
        })

    df_out = pd.DataFrame(rows)
    output_path = data_dir / "train.parquet"
    df_out.to_parquet(output_path, index=False)
    print(f"Created {output_path}: {len(df_out)} samples")


if __name__ == "__main__":
    convert()