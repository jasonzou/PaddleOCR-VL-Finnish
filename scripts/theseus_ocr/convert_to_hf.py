"""
Convert theseus_ocr_dataset to a HuggingFace-compatible dataset.

Two modes (--embed / --no-embed):

  --embed   (default)
    Images are embedded as bytes inside sharded parquet files.
    Shard size is controlled by --shard-size (MB, default 50).
    Output:
      data/train-00000-of-NNNNN.parquet
      ...
      README.md

  --no-embed
    Images are copied to images/ as PNG files.
    All metadata is saved in a single data/train-00000-of-00001.parquet.
    Output:
      data/train-00000-of-00001.parquet
      images/<pdf_stem>_<page>_para<nn>.png
      README.md
"""

import argparse
import io
import json
import os
import shutil
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

DATASET_JSON = "theseus_ocr_dataset/theseus_ocr_dataset.json"
CROPS_ROOT   = "theseus_ocr_dataset"
HF_OUT       = "theseus_ocr_hf"


# ── helpers ──────────────────────────────────────────────────────────────────

def _image_bytes(src: str) -> bytes | None:
    try:
        with Image.open(src) as img:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
    except Exception as e:
        print(f"⚠️  {src}: {e}")
        return None


def _copy_image(src: str, dst: str) -> bool:
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        print(f"⚠️  {src}: {e}")
        return False


def _crop_dest_name(rec: dict) -> str:
    """Stable flat filename: <pdf_stem>_<page>_para<nn>.png"""
    pdf_stem = os.path.splitext(rec["pdf_file"])[0]
    base = os.path.basename(rec["image"])               # p3_para12.png
    parts = base.replace(".png", "").split("_")         # ["p3", "para12"]
    page  = parts[0][1:]                                # "3"
    para  = parts[1][4:].zfill(2)                       # "12"
    return f"{pdf_stem}_{page}_para{para}.png"


# ── embed mode ────────────────────────────────────────────────────────────────

def write_embedded(records: list[dict], out_dir: str, shard_mb: int) -> int:
    os.makedirs(out_dir, exist_ok=True)
    max_bytes  = shard_mb * 1024 * 1024

    shards: list[pa.Table] = []
    pdf_files, pages, texts, images = [], [], [], []
    shard_bytes = 0

    def flush():
        if pdf_files:
            shards.append(pa.table({
                "pdf_file": pa.array(pdf_files, pa.string()),
                "page":     pa.array(pages,     pa.int32()),
                "text":     pa.array(texts,     pa.string()),
                "image":    pa.array(images,    pa.struct([
                                pa.field("bytes", pa.binary()),
                                pa.field("path",  pa.string()),
                            ])),
            }))
            pdf_files.clear()
            pages.clear()
            texts.clear()
            images.clear()

    for rec in tqdm(records, desc="Embedding images"):
        src = os.path.join(CROPS_ROOT, rec["image"])
        img_bytes = _image_bytes(src)
        if img_bytes is None:
            continue
        size = len(img_bytes)
        if shard_bytes + size > max_bytes and pdf_files:
            flush()
            shard_bytes = 0
        pdf_files.append(rec["pdf_file"])
        pages.append(rec["page"])
        texts.append(rec["text"])
        images.append({"bytes": img_bytes, "path": _crop_dest_name(rec)})
        shard_bytes += size

    flush()

    total = len(shards)
    rows  = 0
    for i, tbl in enumerate(shards):
        path = os.path.join(out_dir, f"data-{i+1:05d}-of-{total:05d}.parquet")
        pq.write_table(tbl, path, compression="snappy")
        mb = os.path.getsize(path) / 1024 / 1024
        print(f"  Shard {i+1}/{total}: {len(tbl)} rows, {mb:.1f} MB → {path}")
        rows += len(tbl)
    return rows


# ── no-embed mode ─────────────────────────────────────────────────────────────

def write_separate(records: list[dict], out_dir: str) -> int:
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(out_dir,    exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    pdf_files, pages, texts, image_paths = [], [], [], []

    for rec in tqdm(records, desc="Copying images"):
        src      = os.path.join(CROPS_ROOT, rec["image"])
        dst_name = _crop_dest_name(rec)
        dst      = os.path.join(images_dir, dst_name)
        if not _copy_image(src, dst):
            continue
        pdf_files.append(rec["pdf_file"])
        pages.append(rec["page"])
        texts.append(rec["text"])
        image_paths.append(os.path.join("images", dst_name))

    table = pa.table({
        "pdf_file":   pa.array(pdf_files,   pa.string()),
        "page":       pa.array(pages,       pa.int32()),
        "text":       pa.array(texts,       pa.string()),
        "image_path": pa.array(image_paths, pa.string()),
    })
    path = os.path.join(out_dir, "data.parquet")
    pq.write_table(table, path, compression="snappy")
    mb = os.path.getsize(path) / 1024 / 1024
    print(f"  Metadata: {len(table)} rows, {mb:.1f} MB → {path}")
    print(f"  Images  : {len(table)} PNGs → {images_dir}/")
    return len(table)


# ── README ────────────────────────────────────────────────────────────────────

README_TEMPLATE = """\
---
license: cc-by-4.0
language:
- fi
task_categories:
- image-to-text
pretty_name: Theseus Finnish OCR
dataset_info:
  features:
    - name: pdf_file
      dtype: string
    - name: page
      dtype: int32
    - name: text
      dtype: string
    - name: {image_field}
      dtype: {image_dtype}
  splits:
    - name: data
      num_examples: {num_examples}
---

# Theseus Finnish OCR Dataset

Paragraph-level OCR dataset harvested from [Theseus.fi](https://www.theseus.fi),
the Finnish repository of university of applied sciences theses.

Each record is one paragraph crop extracted from a thesis PDF, paired with the
text extracted by `pdfplumber`.

## Image Resolution

Paragraph crops are rendered at **{resolution} DPI** (dots per inch) with 2 px
padding on each side. At {resolution} DPI a standard A4 page is
{page_w} × {page_h} pixels.

## Fields

| Field | Type | Description |
|-------|------|-------------|
| `pdf_file` | string | Source PDF filename |
| `page` | int | Page number (1-indexed) |
| `{image_field}` | {image_dtype} | {image_desc} |
| `text` | string | Extracted paragraph text from `pdfplumber` |

## Storage format

{storage_note}

## Source

Records were harvested via the Theseus OAI-PMH endpoint and PDFs were
downloaded from the DSpace bitstream API.
"""


def write_readme(out_dir: str, num_examples: int, embed: bool,
                 resolution: int = 300) -> None:
    page_w = round(8.27 * resolution)
    page_h = round(11.69 * resolution)

    if embed:
        image_field = "image"
        image_dtype = "image"
        image_desc  = f"Paragraph crop at {resolution} DPI with 2 px padding (embedded bytes)"
        storage_note = (
            "Images are **embedded** as PNG bytes inside the parquet shards "
            "(HuggingFace `Image` feature)."
        )
    else:
        image_field = "image_path"
        image_dtype = "string"
        image_desc  = f"Relative path to PNG crop under `images/` at {resolution} DPI with 2 px padding"
        storage_note = (
            "Images are stored as separate PNG files under `images/`. "
            "The `image_path` column contains the relative path to each file."
        )

    path = os.path.join(out_dir, "README.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(README_TEMPLATE.format(
            num_examples=num_examples,
            resolution=resolution,
            page_w=page_w,
            page_h=page_h,
            image_field=image_field,
            image_dtype=image_dtype,
            image_desc=image_desc,
            storage_note=storage_note,
        ))
    print(f"  README  : {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert theseus_ocr_dataset to HuggingFace format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    embed_group = parser.add_mutually_exclusive_group()
    embed_group.add_argument(
        "--embed", dest="embed", action="store_true", default=True,
        help="Embed images as bytes inside parquet (default)",
    )
    embed_group.add_argument(
        "--no-embed", dest="embed", action="store_false",
        help="Store images as separate PNGs; parquet holds paths only",
    )
    parser.add_argument(
        "--shard-size", "-s",
        type=int, default=50, metavar="MB",
        help="Max parquet shard size in MB (embed mode only)",
    )
    parser.add_argument(
        "--resolution", "-r",
        type=int, default=300, metavar="DPI",
        help="DPI used when crops were rendered (written to README only)",
    )
    parser.add_argument(
        "--output", "-o",
        default=HF_OUT, metavar="DIR",
        help="Output directory",
    )
    args = parser.parse_args()

    with open(DATASET_JSON, encoding="utf-8") as f:
        records = json.load(f)

    mode = "embedded parquet" if args.embed else "separate images + parquet"
    print(f"Converting {len(records)} records → '{args.output}/' ({mode})")
    os.makedirs(args.output, exist_ok=True)

    if args.embed:
        rows = write_embedded(records, args.output, shard_mb=args.shard_size)
    else:
        rows = write_separate(records, args.output)

    write_readme(args.output, num_examples=rows, embed=args.embed,
                 resolution=args.resolution)

    print(f"\nDone — {rows} records.")
    print("Push with:")
    print(f"  huggingface-cli upload caveman273/theseus-finnish-ocr {args.output} --repo-type dataset")


if __name__ == "__main__":
    main()
