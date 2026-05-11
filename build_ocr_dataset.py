"""
OCR dataset builder.

Subcommands:
  hf           Download each caveman273/* HF dataset → data/hf/{dataset}.parquet
               --gen  also generate data/hf_gen/{dataset}/train.jsonl, eval.jsonl, test.jsonl
  digi-natlib  Clone sdrobac/nodalida2017 and build data/digi-natlib/train.parquet
               --gen  also generate data/digi-natlib-gen/train.jsonl, eval.jsonl, test.jsonl
"""

import argparse
import hashlib
import json
import os
import sys
from io import BytesIO
from pathlib import Path

from tqdm import tqdm


# ── shared helpers ────────────────────────────────────────────────────────────

def det_hash(s: str) -> float:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) / (16**32)


# ── subcommand: hf ────────────────────────────────────────────────────────────

HF_DATASETS = [
    "caveman273/aida-handwritten",
    "caveman273/aida-typewritten",
    "caveman273/aida-ship-info",
    "caveman273/fin-13k",
    "caveman273/swe-11k",
]

HF_OUT_DIR  = "./data/hf"
HF_GEN_DIR  = "./data/hf_gen"
HF_PARQUET  = "train.parquet"


def _img_to_bytes(img, PILImage) -> bytes | None:
    """Convert any image representation to PNG bytes."""
    try:
        if isinstance(img, PILImage.Image):
            buf = BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        if isinstance(img, dict) and "bytes" in img and img["bytes"]:
            buf = BytesIO()
            PILImage.open(BytesIO(img["bytes"])).save(buf, format="PNG")
            return buf.getvalue()
        if isinstance(img, (str, Path)) and Path(img).exists():
            buf = BytesIO()
            PILImage.open(img).save(buf, format="PNG")
            return buf.getvalue()
    except Exception as e:
        print(f"    image conversion error: {e}")
    return None


def _write_parquet(sources, file_names, texts, image_bytes_list, parquet):
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({
        "source":    pa.array(sources,           pa.string()),
        "file_name": pa.array(file_names,        pa.string()),
        "text":      pa.array(texts,             pa.string()),
        "image":     pa.array([{"bytes": b, "path": n} for b, n in zip(image_bytes_list, file_names)],
                              pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])),
    })
    pq.write_table(table, parquet, compression="snappy")
    return len(table)


def cmd_hf(args):
    try:
        from datasets import load_dataset
        from PIL import Image as PILImage
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        print(f"Missing dependency: {e}\nRun: pip install datasets pillow pyarrow", file=sys.stderr)
        sys.exit(1)

    out_dir    = Path(args.hf_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds_name in HF_DATASETS:
        prefix  = ds_name.split("/")[-1].replace("-", "_")
        parquet = out_dir / f"{prefix}.parquet"

        if parquet.exists():
            print(f"  [skip] {parquet} already exists")
            continue

        print(f"  Loading {ds_name} ...")
        try:
            ds = load_dataset(ds_name, split="train")
        except Exception as e:
            print(f"  ERROR loading {ds_name}: {e}")
            continue

        img_col  = next((c for c in ("image", "img", "image_path") if c in ds.column_names), None)
        text_col = next((c for c in ("text", "transcript", "label", "ground_truth", "ocr_text") if c in ds.column_names), None)
        name_col = next((c for c in ("file_name", "filename", "id") if c in ds.column_names), None)

        if img_col is None or text_col is None:
            print(f"  WARNING: could not auto-detect columns in {ds_name}. Columns: {ds.column_names}")
            continue

        sources, file_names, texts, image_bytes_list = [], [], [], []
        written = 0
        for i, row in enumerate(tqdm(ds, desc=prefix, unit="img", leave=False)):
            text = row[text_col]
            if not isinstance(text, str) or not text.strip():
                continue
            img_bytes = _img_to_bytes(row[img_col], PILImage)
            if img_bytes is None:
                continue
            fname = row[name_col] if name_col else f"{i:08d}"
            if not str(fname).endswith(".png"):
                fname = f"{fname}.png"
            sources.append(prefix)
            file_names.append(str(fname))
            texts.append(text.strip())
            image_bytes_list.append(img_bytes)
            written += 1

        if not sources:
            print(f"  {ds_name}: no valid rows")
            continue

        n = _write_parquet(sources, file_names, texts, image_bytes_list, parquet)
        mb = parquet.stat().st_size / 1024 / 1024
        print(f"  {ds_name}: {written} rows → {parquet.name} ({mb:.1f} MB)")

    if args.gen:
        for ds_name in HF_DATASETS:
            prefix = ds_name.split("/")[-1].replace("-", "_")
            parquet = out_dir / f"{prefix}.parquet"
            if parquet.exists():
                gen_dir = Path(args.hf_gen_dir) / prefix
                _hf_gen(parquet, gen_dir, args.train_ratio, args.eval_ratio)


def _hf_gen(parquet: Path, gen_dir: Path, train_ratio: float, eval_ratio: float) -> None:
    """Generate train/eval/test JSONL from the merged parquet."""
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        sys.exit(1)

    gen_dir.mkdir(parents=True, exist_ok=True)
    images_dir = gen_dir / "images"
    images_dir.mkdir(exist_ok=True)

    pf = pq.ParquetFile(parquet)
    n = pf.metadata.num_rows
    print(f"  Reading {n} rows from {parquet} ...")

    entries = []
    for batch in tqdm(pf.iter_batches(batch_size=10000), desc="reading rows", unit="batch"):
        batch_dict = batch.to_pydict()
        sources = batch_dict["source"]
        file_names = batch_dict["file_name"]
        texts = batch_dict["text"]
        images_col = batch_dict["image"]
        pdf_files = batch_dict.get("pdf_file", [None] * len(sources))

        for src, fname, text, img_struct, pdf_file in zip(sources, file_names, texts, images_col, pdf_files):
            if not isinstance(text, str) or not text.strip():
                continue
            path_prefix = ""
            if fname and "/" in fname:
                path_prefix = fname.split("/")[0] + "_"
                fname = fname.split("/")[-1]
            img_filename = f"{path_prefix}{fname}"
            img_dest = images_dir / img_filename
            if not img_dest.exists():
                raw = img_struct.get("bytes") if isinstance(img_struct, dict) else img_struct["bytes"]
                img_dest.write_bytes(raw)
            entries.append({
                "messages": [
                    {"role": "user", "content": "<image>OCR:"},
                    {"role": "assistant", "content": text},
                ],
                "images": [f"./images/{img_filename}"],
            })

    entries.sort(key=lambda e: det_hash(e["images"][0]))

    train_end = int(n * train_ratio)
    eval_end = train_end + int(n * eval_ratio)
    splits = {
        "train": entries[:train_end],
        "eval": entries[train_end:eval_end],
        "test": entries[eval_end:],
    }

    train_imgs = {e["images"][0] for e in splits["train"]}
    eval_imgs = {e["images"][0] for e in splits["eval"]}
    test_imgs = {e["images"][0] for e in splits["test"]}

    overlaps = (train_imgs & eval_imgs) | (train_imgs & test_imgs) | (eval_imgs & test_imgs)
    if overlaps:
        print(f"  WARNING: {len(overlaps)} overlaps found, removing from eval/test")
        splits["eval"] = [e for e in splits["eval"] if e["images"][0] not in train_imgs]
        eval_imgs = {e["images"][0] for e in splits["eval"]}
        splits["test"] = [e for e in splits["test"] if e["images"][0] not in train_imgs and e["images"][0] not in eval_imgs]
        test_imgs = {e["images"][0] for e in splits["test"]}
        eval_test_overlaps = eval_imgs & test_imgs
        if eval_test_overlaps:
            splits["test"] = [e for e in splits["test"] if e["images"][0] not in eval_imgs]

    for split_name, rows in splits.items():
        out_path = gen_dir / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {split_name}: {len(rows)} entries → {out_path}")


# ── subcommand: digi-natlib ───────────────────────────────────────────────────

NODALIDA_REPO   = "https://github.com/sdrobac/nodalida2017"
NODALIDA_CLONE  = "./data/nodalida2017"

DN_OUT_DIR = "./data/digi-natlib"
DN_GEN_DIR = "./data/digi-natlib-gen"

# sub-directories inside the cloned repo and their source label
_NODALIDA_SUBDIRS = [
    ("digi",   "data/digi-data"),
    ("natlib", "data/natlib-data"),
]

# which splits to include (all three folders inside each source dir)
_NODALIDA_SPLITS = ("train", "test-dev", "test-test")


def _nodalida_clone(clone_dir: Path) -> None:
    """Shallow-clone or update the nodalida2017 repo."""
    import subprocess
    if (clone_dir / ".git").exists():
        print(f"  [git] Updating {clone_dir} ...")
        subprocess.run(["git", "-C", str(clone_dir), "pull", "--ff-only", "-q"],
                       check=True)
    else:
        clone_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"  [git] Cloning {NODALIDA_REPO} → {clone_dir} ...")
        subprocess.run(["git", "clone", "--depth=1", "-q", NODALIDA_REPO, str(clone_dir)],
                       check=True)


def _nodalida_iter(clone_dir: Path):
    """Yield (source_label, file_name, text, png_bytes) for every .bin.png/.gt.txt pair."""
    for source_label, sub in _NODALIDA_SUBDIRS:
        sub_root = clone_dir / sub
        if not sub_root.is_dir():
            print(f"  [skip] {sub_root} not found in clone")
            continue
        for split in _NODALIDA_SPLITS:
            split_dir = sub_root / split
            if not split_dir.is_dir():
                continue
            for img_path in sorted(split_dir.rglob("*.bin.png")):
                gt_path = img_path.parent / (img_path.name[: -len(".bin.png")] + ".gt.txt")
                if not gt_path.exists():
                    continue
                text = gt_path.read_text(encoding="utf-8").strip()
                if not text:
                    continue
                # build a unique, collision-free filename: source__split__stem.png
                stem     = img_path.stem[: -len(".bin")]   # strip the .bin part
                fname    = f"{source_label}__{split}__{stem}.png"
                yield source_label, fname, text, img_path.read_bytes()


def cmd_digi_natlib(args):
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        print(f"Missing dependency: {e}\nRun: pip install pyarrow", file=sys.stderr)
        sys.exit(1)

    out_dir    = Path(args.dn_out_dir)
    parquet    = out_dir / HF_PARQUET
    clone_dir  = Path(args.nodalida_clone)
    out_dir.mkdir(parents=True, exist_ok=True)

    if parquet.exists():
        print(f"  [skip] {parquet} already exists (delete to re-build)")
    else:
        _nodalida_clone(clone_dir)

        sources, file_names, texts, image_bytes_list = [], [], [], []
        for source_label, fname, text, png_bytes in tqdm(
                _nodalida_iter(clone_dir), desc="nodalida2017", unit="img"):
            sources.append(source_label)
            file_names.append(fname)
            texts.append(text)
            image_bytes_list.append(png_bytes)

        if not sources:
            print("  ERROR: no image/text pairs found in clone — check repo layout")
            return

        digi_n   = sum(1 for s in sources if s == "digi")
        natlib_n = sum(1 for s in sources if s == "natlib")
        print(f"  digi: {digi_n} rows,  natlib: {natlib_n} rows")

        table = pa.table({
            "source":    pa.array(sources,    pa.string()),
            "file_name": pa.array(file_names, pa.string()),
            "text":      pa.array(texts,      pa.string()),
            "image":     pa.array(
                [{"bytes": b, "path": n} for b, n in zip(image_bytes_list, file_names)],
                pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())]),
            ),
        })
        pq.write_table(table, parquet, compression="snappy")
        mb = parquet.stat().st_size / 1024 / 1024
        print(f"\n  Wrote {len(table)} rows → {parquet} ({mb:.1f} MB)")

    if args.gen:
        _hf_gen(parquet, Path(args.dn_gen_dir), args.train_ratio, args.eval_ratio)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "./data/ocr_merged"))

    sub = parser.add_subparsers(dest="cmd", required=True)

    # hf
    p_hf = sub.add_parser("hf", parents=[shared], help="Merge HF datasets → data/hf/train.parquet")
    p_hf.add_argument("--hf-out-dir",  default=os.environ.get("HF_OUT_DIR",  HF_OUT_DIR))
    p_hf.add_argument("--hf-gen-dir",  default=os.environ.get("HF_GEN_DIR",  HF_GEN_DIR))
    p_hf.add_argument("--gen",         action="store_true", help="Also generate train/eval/test JSONL in --hf-gen-dir")
    p_hf.add_argument("--train-ratio", type=float, default=float(os.environ.get("TRAIN_RATIO", "0.90")))
    p_hf.add_argument("--eval-ratio",  type=float, default=float(os.environ.get("EVAL_RATIO",  "0.05")))
    p_hf.set_defaults(func=cmd_hf)

    # digi-natlib
    p_dn = sub.add_parser("digi-natlib", parents=[shared], help="Download sdrobac/nodalida2017 and build data/digi-natlib/train.parquet")
    p_dn.add_argument("--dn-out-dir",     default=os.environ.get("DN_OUT_DIR",     DN_OUT_DIR))
    p_dn.add_argument("--dn-gen-dir",     default=os.environ.get("DN_GEN_DIR",     DN_GEN_DIR))
    p_dn.add_argument("--nodalida-clone", default=os.environ.get("NODALIDA_CLONE", NODALIDA_CLONE))
    p_dn.add_argument("--gen",            action="store_true", help="Also generate train/eval/test JSONL in --dn-gen-dir")
    p_dn.add_argument("--train-ratio",    type=float, default=float(os.environ.get("TRAIN_RATIO", "0.90")))
    p_dn.add_argument("--eval-ratio",     type=float, default=float(os.environ.get("EVAL_RATIO",  "0.05")))
    p_dn.set_defaults(func=cmd_digi_natlib)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
