import os
import io
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

IMAGES_DIR = "images"
OUTPUT_DIR = "./"

SPLITS = {
    "train": "train_typewritten_best.txt",
    "test": "test_typewritten_best.txt",
    "validation": "val_typewritten_best.txt",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

SCHEMA = pa.schema([
    pa.field("image", pa.struct([
        pa.field("bytes", pa.binary()),
        pa.field("path", pa.string()),
    ])),
    pa.field("text", pa.string()),
    pa.field("file_name", pa.string()),
])


def image_bytes(path):
    with Image.open(path) as img:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


for split, txt_file in SPLITS.items():
    rows = {"image": [], "text": [], "file_name": []}
    missing = 0

    with open(txt_file) as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) < 2:
                continue
            filename, text = parts
            img_path = os.path.join(IMAGES_DIR, filename)
            if not os.path.exists(img_path):
                missing += 1
                continue
            rows["image"].append({"bytes": image_bytes(img_path), "path": filename})
            rows["text"].append(text)
            rows["file_name"].append(filename)

    table = pa.table(
        {
            "image": pa.array(rows["image"], type=SCHEMA.field("image").type),
            "text": pa.array(rows["text"], type=pa.string()),
            "file_name": pa.array(rows["file_name"], type=pa.string()),
        }
    )

    out_path = os.path.join(OUTPUT_DIR, f"{split}.parquet")
    pq.write_table(table, out_path, compression="snappy")
    print(f"{split}: {len(rows['text'])} rows written, {missing} missing -> {out_path}")
