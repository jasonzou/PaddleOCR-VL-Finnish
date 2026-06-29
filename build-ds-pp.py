"""
OCR dataset builder.

Parquet schema (all subcommands): source, file_name, text, type, image
  where type ∈ {"text_line", "paragraph", "page"} describes the image granularity.

Subcommands:
  hf           Download each caveman273/* HF dataset → data/hf/{dataset}.parquet  (type=text_line)
               --gen  also generate data/hf_gen/{dataset}/train.jsonl, eval.jsonl, test.jsonl
  digi-natlib  Clone sdrobac/nodalida2017 and build data/digi-natlib/train.parquet  (type=text_line)
               --gen  also generate data/digi-natlib-gen/train.jsonl, eval.jsonl, test.jsonl
  synth        Render synthetic Finnish text images → data/synth-fonts/dataset.parquet  (type=paragraph)
  theseus      Harvest handles via OAI-PMH, download PDFs, extract page + paragraph crops
               → data/theseus-dataset/train.parquet  (type=page, paragraph)
  theseus-pdfs Extract page + paragraph crops from already-downloaded PDFs in --theseus-pdf-dir
  nlf          Extract page + paragraph crops from NLF OCR ground-truth (ALTO XML + TIFF)
               → data/nlf_ocr/train.parquet  (type=page, paragraph)
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import textwrap
from io import BytesIO
from pathlib import Path

from tqdm import tqdm


# ── shared helpers ────────────────────────────────────────────────────────────

def det_hash(s: str) -> float:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) / (16**32)


# ── subcommand: hf ────────────────────────────────────────────────────────────

HF_PARQUET = "train.parquet"

HF_DATASETS = [
    "caveman273/aida-handwritten",
    "caveman273/aida-typewritten",
    "caveman273/aida-ship-info",
    "caveman273/fin-13k",
    "caveman273/swe-11k",
]

HF_OUT_DIR  = "./data/hf"
HF_GEN_DIR  = "./data/hf_gen"


def _img_to_bytes(img, PILImage):
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


def _merge_parquets(parquet_files, out_path):
    """Concatenate multiple parquet files into one."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    tables = []
    for pf in parquet_files:
        if pf.exists():
            tables.append(pq.read_table(pf))
    if not tables:
        return
    merged = pa.concat_tables(tables)
    pq.write_table(merged, out_path, compression="snappy")
    print(f"  Merged {len(tables)} parquets ({len(merged)} rows) → {out_path.name}")


def _write_parquet(sources, file_names, texts, image_bytes_list, parquet, types="text_line"):
    """Write a parquet manifest.

    *types* may be a single string (broadcast to every row) or a per-row list
    of values from {"text_line", "paragraph", "page"} describing the image
    granularity of each sample.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    if isinstance(types, str):
        types = [types] * len(texts)
    table = pa.table({
        "source":    pa.array(sources,           pa.string()),
        "file_name": pa.array(file_names,        pa.string()),
        "text":      pa.array(texts,             pa.string()),
        "type":      pa.array(types,             pa.string()),
        "image":     pa.array([{"bytes": b, "path": n} for b, n in zip(image_bytes_list, file_names)],
                              pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])),
    })
    pq.write_table(table, parquet, compression="snappy")
    return len(table)


SPLITS = ["train", "test", "validation"]
PARQUET_NAMES = {
    "train": "{prefix}_train.parquet",
    "test": "{prefix}_test.parquet",
    "validation": "{prefix}_validation.parquet",
}
JSONL_NAMES = {"train": "train.jsonl", "test": "test.jsonl", "validation": "eval.jsonl"}


def cmd_hf(args):
    try:
        from datasets import load_dataset
        from PIL import Image as PILImage
    except ImportError as e:
        print(f"Missing dependency: {e}\nRun: pip install datasets pillow", file=sys.stderr)
        sys.exit(1)

    if args.hf_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    out_dir    = Path(args.hf_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds_name in HF_DATASETS:
        prefix  = ds_name.split("/")[-1].replace("-", "_")
        print(f"  Checking {ds_name} ...")
        try:
            ds_info = load_dataset(ds_name)
        except Exception as e:
            print(f"  ERROR loading {ds_name}: {e}")
            continue

        available_splits = list(ds_info.keys())
        for split in SPLITS:
            if split not in available_splits:
                continue
            parquet = out_dir / PARQUET_NAMES[split].format(prefix=prefix)
            if parquet.exists():
                print(f"  [skip] {parquet.name} already exists")
                continue

            print(f"  Loading {ds_name} ({split}) ...")
            try:
                ds = load_dataset(ds_name, split=split)
            except Exception as e:
                print(f"  ERROR loading {ds_name}/{split}: {e}")
                continue

            img_col  = next((c for c in ("image", "img", "image_path") if c in ds.column_names), None)
            text_col = next((c for c in ("text", "transcript", "label", "ground_truth", "ocr_text") if c in ds.column_names), None)
            name_col = next((c for c in ("file_name", "filename", "id") if c in ds.column_names), None)

            if img_col is None or text_col is None:
                print(f"  WARNING: could not auto-detect columns in {ds_name}. Columns: {ds.column_names}")
                continue

            sources, file_names, texts, image_bytes_list = [], [], [], []
            written = 0
            for i, row in enumerate(tqdm(ds, desc=f"{prefix}_{split}", unit="img", leave=False)):
                text = row[text_col]
                if not isinstance(text, str) or not text.strip():
                    continue
                img_val = row[img_col]
                if isinstance(img_val, (str, Path)) and not Path(img_val).exists():
                    continue
                img_bytes = _img_to_bytes(img_val, PILImage)
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
                print(f"  {ds_name}/{split}: no valid rows")
                continue

            n = _write_parquet(sources, file_names, texts, image_bytes_list, parquet, types="text_line")
            mb = parquet.stat().st_size / 1024 / 1024
            print(f"  {ds_name}/{split}: {written} rows → {parquet.name} ({mb:.1f} MB)")

    if args.gen:
        for ds_name in HF_DATASETS:
            prefix = ds_name.split("/")[-1].replace("-", "_")
            gen_dir = Path(args.hf_gen_dir) / prefix

            existing_parquets = [
                out_dir / PARQUET_NAMES[split].format(prefix=prefix)
                for split in SPLITS
                if (out_dir / PARQUET_NAMES[split].format(prefix=prefix)).exists()
            ]

            if not existing_parquets:
                continue

            if args.kfolds and args.kfolds >= 2:
                # Merge all splits into one, then k-fold from scratch
                merged_parquet = out_dir / f"{prefix}_merged.parquet"
                if not merged_parquet.exists():
                    _merge_parquets(existing_parquets, merged_parquet)
                _hf_gen(merged_parquet, gen_dir, args.train_ratio, args.eval_ratio, args.kfolds)
            elif len(existing_parquets) == 1:
                parquet = existing_parquets[0]
                print(f"  {prefix}: single split ({parquet.name}), generating train/eval/test")
                _hf_gen(parquet, gen_dir, args.train_ratio, args.eval_ratio, 0)
            else:
                for split in SPLITS:
                    parquet = out_dir / PARQUET_NAMES[split].format(prefix=prefix)
                    if parquet.exists():
                        _hf_gen_direct(parquet, gen_dir, JSONL_NAMES[split], split)


def _read_parquet_rows(parquet: Path):
    """Yield row dicts from a parquet file, handling nested struct columns.

    Each dict has keys: source, file_name, text, image (bytes),
    and optionally pdf_file, page, type.

    Uses fastparquet because pyarrow 24 on Python 3.14 cannot read
    nested struct columns (``ArrowNotImplementedError``).
    """
    import fastparquet as fp

    pf = fp.ParquetFile(str(parquet))
    n = pf.count()
    df = pf.to_pandas()

    for _, row in df.iterrows():
        d = {}
        d["source"] = row.get("source")
        d["file_name"] = row.get("file_name")
        d["text"] = row.get("text")
        d["type"] = row.get("type")
        if "pdf_file" in df.columns:
            d["pdf_file"] = row.get("pdf_file")
        if "page" in df.columns:
            d["page"] = row.get("page")
        # fastparquet flattens struct → image.bytes / image.path
        if "image.bytes" in df.columns:
            d["image_bytes"] = row["image.bytes"]
        elif "image" in df.columns:
            img = row["image"]
            d["image_bytes"] = img["bytes"] if isinstance(img, dict) else img
        else:
            d["image_bytes"] = None
        yield d


def _split_entries(entries, n, train_ratio, eval_ratio):
    """Deterministic hash-based train/eval/test split."""
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

    return splits


def _write_jsonl(entries, out_path: Path, label: str = "") -> int:
    """Write entries to a JSONL file. Returns count written."""
    with open(out_path, "w", encoding="utf-8") as f:
        for row in entries:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    count = len(entries)
    tag = f"{label}: " if label else ""
    print(f"  {tag}{count} entries → {out_path}")
    return count


def _hf_gen(parquet: Path, gen_dir: Path, train_ratio: float, eval_ratio: float,
            kfolds: int = 0, name_func=None) -> None:
    """Generate train/eval/test JSONL from the merged parquet with train/eval/test split.

    When *kfolds* > 0, produces ``fold_0/`` … ``fold_{k-1}/`` subdirectories
    under *gen_dir*.  Each fold uses 1/k of the data as eval and the rest as
    train; the test set is shared across all folds (the last ``eval_ratio``
    fraction of the data).

    *name_func*, if given, is called with each row dict and must return the
    output image filename (without directory).  When ``None``, the filename is
    derived from the ``file_name`` column as before.
    """
    gen_dir.mkdir(parents=True, exist_ok=True)
    images_dir = gen_dir / "images"
    images_dir.mkdir(exist_ok=True)

    rows = list(_read_parquet_rows(parquet))
    n = len(rows)
    print(f"  Reading {n} rows from {parquet} ...")

    entries = []
    for row in tqdm(rows, desc="processing rows", unit="row"):
        text = row["text"]
        if not isinstance(text, str) or not text.strip():
            continue
        if name_func:
            img_filename = name_func(row)
        else:
            fname = row["file_name"]
            path_prefix = ""
            if fname and "/" in fname:
                path_prefix = fname.split("/")[0] + "_"
                fname = fname.split("/")[-1]
            img_filename = f"{path_prefix}{fname}"
        img_dest = images_dir / img_filename
        if not img_dest.exists() and row.get("image_bytes"):
            img_dest.write_bytes(row["image_bytes"])
        entries.append({
            "messages": [
                {"role": "user", "content": "<image>OCR:"},
                {"role": "assistant", "content": text},
            ],
            "images": [f"./images/{img_filename}"],
        })

    entries.sort(key=lambda e: det_hash(e["images"][0]))

    if kfolds and kfolds >= 2:
        _write_kfolds(entries, gen_dir, images_dir, kfolds, train_ratio)
    else:
        splits = _split_entries(entries, len(entries), train_ratio, eval_ratio)
        for split_name, rows in splits.items():
            _write_jsonl(rows, gen_dir / f"{split_name}.jsonl", split_name)


def _write_kfolds(entries, gen_dir: Path, images_dir: Path, k: int, train_ratio: float) -> None:
    """Write k-fold cross-validation splits.

    Layout::

        gen_dir/
          images/
          fold_0_train.jsonl  fold_0_eval.jsonl
          fold_1_train.jsonl  fold_1_eval.jsonl
          ...
          test.jsonl

    * The test set is fixed: the last ``(1 - train_ratio)`` fraction of entries.
    * The train+eval pool is everything before the test set — test **never**
      participates in any fold.
    * Each fold uses 1/k of that pool as eval and the rest as train.
    """
    n = len(entries)
    rng = random.Random(42)
    shuffled = list(entries)
    rng.shuffle(shuffled)

    # 1. Carve out test set: last (1 - train_ratio) of shuffled entries.
    # 2. Pool is everything else. Each fold gets pool // k eval entries;
    #    the last fold gets the remainder.
    test_size = n - int(n * train_ratio)
    pool = shuffled[:n - test_size]
    test = shuffled[n - test_size:]

    assert len(pool) + len(test) == n
    test_imgs = {e["images"][0] for e in test}

    fold_size = len(pool) // k
    print(f"  k-fold: k={k}, pool={len(pool)}, test={len(test)}, fold_eval={fold_size}")

    for fold_idx in range(k):
        eval_start = fold_idx * fold_size
        eval_end = eval_start + fold_size if fold_idx < k - 1 else len(pool)

        eval_entries = pool[eval_start:eval_end]
        train_entries = pool[:eval_start] + pool[eval_end:]

        train_imgs = {e["images"][0] for e in train_entries}
        eval_entries = [e for e in eval_entries if e["images"][0] not in train_imgs]

        train_eval_imgs = train_imgs | {e["images"][0] for e in eval_entries}
        assert not (test_imgs & train_eval_imgs), f"fold_{fold_idx}: test images leaked into train/eval"

        _write_jsonl(train_entries, gen_dir / f"fold_{fold_idx}_train.jsonl", f"fold_{fold_idx}/train")
        _write_jsonl(eval_entries, gen_dir / f"fold_{fold_idx}_eval.jsonl", f"fold_{fold_idx}/eval")

    _write_jsonl(test, gen_dir / "test.jsonl", "test")


def _hf_gen_direct(parquet: Path, gen_dir: Path, out_filename: str, split_name: str = None) -> None:
    """Generate a single JSONL directly from parquet (no split)."""
    gen_dir.mkdir(parents=True, exist_ok=True)
    images_dir = gen_dir / "images"
    images_dir.mkdir(exist_ok=True)

    rows = list(_read_parquet_rows(parquet))
    n = len(rows)
    print(f"  Reading {n} rows from {parquet} ...")

    out_path = gen_dir / out_filename
    with open(out_path, "w", encoding="utf-8") as f:
        for row in tqdm(rows, desc="writing rows", unit="row"):
            text = row["text"]
            fname = row["file_name"]
            if not isinstance(text, str) or not text.strip():
                continue
            path_prefix = ""
            if fname and "/" in fname:
                path_prefix = fname.split("/")[0] + "_"
                fname = fname.split("/")[-1]
            img_filename = f"{path_prefix}{fname}"
            img_dest = images_dir / img_filename
            if not img_dest.exists() and row.get("image_bytes"):
                img_dest.write_bytes(row["image_bytes"])
            entry = {
                "messages": [
                    {"role": "user", "content": "<image>OCR:"},
                    {"role": "assistant", "content": text},
                ],
                "images": [f"./images/{img_filename}"],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"  Wrote {n} entries → {out_path}")


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
        import pyarrow  # noqa: F401  (fail-fast probe; _write_parquet does the real work)
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

        n = _write_parquet(sources, file_names, texts, image_bytes_list, parquet, types="text_line")
        mb = parquet.stat().st_size / 1024 / 1024
        print(f"\n  Wrote {n} rows → {parquet} ({mb:.1f} MB)")

    if args.gen:
        _hf_gen(parquet, Path(args.dn_gen_dir), args.train_ratio, args.eval_ratio, args.kfolds)


SYNTH_OUT_DIR           = "./data/synth-fonts"
SYNTH_TEXT_FILE         = "./data/finnish_corpus.txt"
SYNTH_FONT_DIR          = "finnish-fonts"
SYNTH_FONT_SIZES        = [24, 28, 32, 36, 40, 48]
SYNTH_SAMPLES_DEFAULT   = 20000
SYNTH_SINGLE_LINE_RATIO = 0.01     # cap: at most 1% of samples may be single-line texts
SYNTH_HF_TOKEN_FILE     = ".hf_token"

# Finnish-NLP HF datasets used to build the text corpus (streaming, up to 10k rows each).
TOP10_HF_FINNISH = [
    "Finnish-NLP/CulturaX_fi_cleaned",
    "Finnish-NLP/Culturax_Finnish_fineweb_edu_predicted",
    "Finnish-NLP/Reddit_Finnish_fineweb_edu_predicted",
    "Finnish-NLP/Fineweb2_fi_edu_score_topic_classified",
    "Finnish-NLP/swe_Finepdfs_edu_scores",
    "Finnish-NLP/isl_Finepdfs_edu_scores",
    "Finnish-NLP/HPLT_1.2_fi_cleaned",
    "Finnish-NLP/smol_datasets_translated_magpie_ultra",
    "Finnish-NLP/lumi_poro2_fi_quality_preds",
    "Finnish-NLP/isl_Fineweb2_edu_scores",
    "Finnish-NLP/OrcaAgentInstruct-mcq",
    "Finnish-NLP/mc4_fi_cleaned",
]


# ── subcommand: synth ──────────────────────────────────────────────────────────

def _synth_setup_hf_auth():
    """Authenticate to the HF Hub for corpus streaming.

    Token resolution order: $HF_TOKEN env var first, then SYNTH_HF_TOKEN_FILE.
    Warns and continues unauthenticated if neither is available.
    """
    try:
        from huggingface_hub import login as hf_login
    except ImportError:
        print("  WARNING: huggingface_hub not installed — continuing unauthenticated")
        return
    token = os.environ.get("HF_TOKEN")
    src = "env"
    if not token:
        token_path = Path(SYNTH_HF_TOKEN_FILE)
        if token_path.exists():
            token = token_path.read_text(encoding="utf-8").strip()
            src = str(token_path)
    if not token:
        print(f"  WARNING: no HF token in $HF_TOKEN or {SYNTH_HF_TOKEN_FILE} — continuing unauthenticated")
        return
    os.environ["HF_TOKEN"] = token
    hf_login(token=token, add_to_git_credential=False)
    print(f"  HF login OK (token from {src})")


def _synth_generate_corpus():
    """Stream Finnish text from HF datasets and cache it to SYNTH_TEXT_FILE.

    Skips any source that fails (network/auth/rate-limit) and proceeds with the rest.
    Does NOT write the cache file when nothing was collected, so a later run
    regenerates. Returns the collected texts ([] if every source failed).
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        print(f"Missing dependency: {e}\nRun: pip install datasets", file=sys.stderr)
        sys.exit(1)

    texts = []
    for name in TOP10_HF_FINNISH:
        try:
            ds = load_dataset(name, split="train", streaming=True)
            for i, row in enumerate(ds):
                if i >= 10000:
                    break
                text = row.get("text") or row.get("content") or row.get("html", "")
                if text and len(text.strip()) > 50:
                    texts.append(text.strip())
            print(f"  {name}: {len(texts)} texts")
        except Exception as e:
            print(f"  {name}: {e}")

    random.shuffle(texts)
    print(f"  Total: {len(texts)} texts")

    if not texts:
        return texts

    with open(SYNTH_TEXT_FILE, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")
    return texts


def _synth_is_valid_text(text):
    """Reject texts shorter than 50 chars or with >50% symbols/digits/punctuation."""
    if len(text) < 50:
        return False
    symbol_number_count = len(re.findall(r'[0-9!@#$%^&*()_+\-=\[\]{}|;:\'",.<>?/\\`~]', text))
    symbol_ratio = symbol_number_count / len(text) if len(text) > 0 else 0
    if symbol_ratio > 0.5:
        return False
    return True


def _synth_load_corpus():
    """Load cached corpus from SYNTH_TEXT_FILE, keeping only valid lines."""
    with open(SYNTH_TEXT_FILE, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if _synth_is_valid_text(line.strip())]


def _synth_load_fonts():
    """Recursively collect all .ttf/.otf files under SYNTH_FONT_DIR."""
    font_files = []
    for ext in ("*.ttf", "*.otf"):
        font_files.extend(Path(SYNTH_FONT_DIR).rglob(ext))
    if not font_files:
        raise RuntimeError(f"No fonts found in {SYNTH_FONT_DIR}")
    return font_files


def _synth_generate_image(text, font_path, font_size, width=None):
    """Render *text* to a grayscale PNG.

    Text is wrapped to *width* cols (random in [45, 80] if None), placed with
    random padding/spacing on a near-white background with near-black ink.
    Returns a PIL "L" image sized to fit the wrapped lines. Pass an explicit
    *width* when the caller has already decided the layout (e.g. to enforce a
    single-line cap upstream).
    """
    from PIL import ImageFont, ImageDraw, Image
    font = ImageFont.truetype(font_path, font_size)
    bg_color = random.randint(235, 255)
    fg_color = random.randint(0, 40)

    if width is None:
        width = random.randint(45, 80)
    wrapped = textwrap.fill(text, width=width)
    lines = wrapped.split("\n")

    line_widths = []
    line_heights = []
    for line in lines:
        bbox = font.getbbox(line)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])

    max_line_width = max(line_widths)
    line_spacing = random.randint(2, 10)
    total_text_height = sum(line_heights) + line_spacing * (len(lines) - 1)

    padding = random.randint(10, 30)
    img_width = max_line_width + padding * 2 + 20
    img_height = total_text_height + padding * 2 + 15

    image = Image.new("L", (img_width, img_height), color=bg_color)
    draw = ImageDraw.Draw(image)

    y = padding
    for line in lines:
        draw.text((padding, y), line, font=font, fill=fg_color)
        bbox = font.getbbox(line)
        y += (bbox[3] - bbox[1]) + line_spacing

    return image


def cmd_synth(args):
    """Generate synthetic Finnish OCR samples.

    Outputs (under --synth-out-dir):
      images/*.png       — rendered page images
      dataset.parquet    — manifest with source/file_name/text/type="paragraph"/image(bytes) —
                           same schema as the hf and digi-natlib subcommands,
                           so it can be fed to _hf_gen for k-fold/ratio splits.
      train/eval/test.jsonl — ready-to-train chat-formatted splits.
    """
    try:
        from PIL import Image as PILImage  # noqa: F401  (validates pillow available)
    except ImportError as e:
        print(f"Missing dependency: {e}\nRun: pip install pillow", file=sys.stderr)
        sys.exit(1)

    _synth_setup_hf_auth()

    if args.hf_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    out_dir = Path(args.synth_out_dir)
    images_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(exist_ok=True)

    # Build corpus if missing or empty (a previous run may have produced nothing).
    text_file = Path(SYNTH_TEXT_FILE)
    if not text_file.exists() or text_file.stat().st_size == 0:
        print("  Generating corpus...")
        if not _synth_generate_corpus():
            raise RuntimeError("Corpus generation produced 0 texts. Check HF access / sources.")

    corpus = _synth_load_corpus()
    if not corpus:
        raise RuntimeError(f"{SYNTH_TEXT_FILE} is empty or all lines failed validation.")
    font_files = _synth_load_fonts()

    samples = args.samples
    print(f"  Loaded {len(corpus)} texts, {len(font_files)} fonts → {samples} samples")

    single_line_cap = max(1, int(samples * args.single_line_ratio))
    single_line_count = 0

    entries = []
    pq_sources, pq_file_names, pq_texts, pq_image_bytes, pq_types = [], [], [], [], []
    valid_idx = 0
    while valid_idx < samples:
        text = random.choice(corpus)
        font_path = str(random.choice(font_files))
        font_size = random.choice(SYNTH_FONT_SIZES)
        width = random.randint(45, 80)

        # Reject single-line candidates once the cap is reached. Bounded retry
        # so we never spin forever if the corpus is dominated by short texts.
        wrapped = textwrap.fill(text, width=width)
        if ("\n" not in wrapped) and single_line_count >= single_line_cap:
            for _ in range(50):
                text = random.choice(corpus)
                width = random.randint(45, 80)
                wrapped = textwrap.fill(text, width=width)
                if "\n" in wrapped:
                    break
        is_single = "\n" not in wrapped

        image = _synth_generate_image(text, font_path, font_size, width=width)
        fname = f"{valid_idx:08d}.png"
        buf = BytesIO()
        image.save(buf, format="PNG")
        img_bytes = buf.getvalue()
        (images_dir / fname).write_bytes(img_bytes)

        # Ground-truth label: multi-line paragraphs keep the visual \n line
        # breaks so the label matches the rendered image; single-line is
        # unchanged (wrapped == text in that case).
        label_text = wrapped

        # Collect for the parquet manifest (same schema as hf/digi-natlib).
        # Per-row type: single-line → text_line, multi-line → paragraph.
        pq_sources.append("synth")
        pq_file_names.append(fname)
        pq_texts.append(label_text)
        pq_image_bytes.append(img_bytes)
        pq_types.append("text_line" if is_single else "paragraph")

        entries.append({
            "messages": [
                {"role": "user", "content": "<image>OCR:"},
                {"role": "assistant", "content": label_text},
            ],
            "images": [f"./images/{fname}"],
        })
        if is_single:
            single_line_count += 1
        valid_idx += 1

        if valid_idx % 1000 == 0:
            print(f"  Generated {valid_idx}/{samples} (single-line so far: {single_line_count})")

    # Write parquet manifest: data/synth-fonts/dataset.parquet (consumable by _hf_gen,
    # same source/file_name/text/image schema as the hf and digi-natlib subcommands).
    parquet = out_dir / "dataset.parquet"
    _write_parquet(pq_sources, pq_file_names, pq_texts, pq_image_bytes, parquet, types=pq_types)
    mb = parquet.stat().st_size / 1024 / 1024
    print(f"  Wrote {len(pq_texts)} rows → {parquet} ({mb:.1f} MB)")

    # Train/eval/test split (unique image paths per entry ⇒ no overlap possible).
    train_size = int(samples * (1 - args.eval_ratio - args.test_ratio))
    eval_size = int(samples * args.eval_ratio)
    train_entries = entries[:train_size]
    eval_entries = entries[train_size:train_size + eval_size]
    test_entries = entries[train_size + eval_size:]

    for split_name, split_entries in [("train", train_entries), ("eval", eval_entries), ("test", test_entries)]:
        out_path = out_dir / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for row in split_entries:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {split_name}: {len(split_entries)} entries → {out_path}")

    print(f"  Done. Dataset saved to {out_dir}")
    print(f"  single-line: {single_line_count}/{samples} (cap was {single_line_cap})")


# ── subcommand: theseus ───────────────────────────────────────────────────────

THESEUS_OAI_URL  = "https://www.theseus.fi/oai/request"
THESEUS_PDF_DIR  = "./data/theseus-pdfs"
THESEUS_DATA_DIR = "./data/theseus-dataset"
THESEUS_GEN_DIR  = "./data/theseus-gen"
THESEUS_DPI      = 300


class _BitstreamFinder:
    from html.parser import HTMLParser as _HTMLParser

    class _Parser(_HTMLParser):
        def __init__(self):
            super().__init__()
            self.url = None
        def handle_starttag(self, tag, attrs):
            if self.url:
                return
            href = dict(attrs).get("href") or ""
            path = href.split("?")[0]
            if "/bitstream/" in path and path.lower().endswith(".pdf"):
                self.url = href.split("&")[0]

    @staticmethod
    def find(html: str):
        p = _BitstreamFinder._Parser()
        p.feed(html)
        return p.url


def _theseus_harvest(oai_url: str, num_records: int):
    """Return up to num_records Theseus handle URLs via OAI-PMH."""
    try:
        import re
        import time
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  Missing dependency: playwright\n  Run: pip install playwright && playwright install chromium", file=sys.stderr)
        sys.exit(1)

    import platform
    system = platform.system()
    if system == "Darwin":
        chrome_paths = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/usr/bin/google-chrome",
            "/usr/bin/chromium",
        ]
    else:
        chrome_paths = [
            "/usr/bin/google-chrome",
            "/usr/bin/chromium",
            "/usr/bin/chromium-browser",
        ]

    chrome_executable = None
    for path in chrome_paths:
        if Path(path).exists():
            chrome_executable = path
            break

    if chrome_executable:
        print(f"  Using Chrome at {chrome_executable}")
    else:
        print("  Chrome not found, using Playwright's bundled Chromium")

    print(f"  Harvesting up to {num_records} records from {oai_url} ...")

    handles = []
    resumption_token = None

    with sync_playwright() as p:
        launch_opts = {"headless": True}
        if chrome_executable:
            launch_opts["executable_path"] = chrome_executable
        browser = p.chromium.launch(**launch_opts)
        page = browser.new_page()
        page.set_default_timeout(30000)

        while len(handles) < num_records:
            params = {"verb": "ListRecords", "metadataPrefix": "oai_dc"}
            if resumption_token:
                params["resumptionToken"] = resumption_token

            url = oai_url + "?" + "&".join(f"{k}={v}" for k, v in params.items())
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            time.sleep(3)
            text = page.content()

            print(f"  DEBUG cloudflare={'cloudflare' in text.lower()} body[:200]={text[:200]}")

            if "cloudflare" in text.lower() or "just a moment" in text.lower():
                print("  Cloudflare challenge detected, waiting 10s...")
                time.sleep(10)
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                time.sleep(5)
                text = page.content()
                print(f"  DEBUG retry cloudflare={'cloudflare' in text.lower()} body[:200]={text[:200]}")

            if "<OAI-PMH" not in text and "<errors" not in text:
                print(f"  No OAI-PMH XML found, skipping page")
                break

            for ident in re.findall(r"<dc:identifier[^>]*>([^<]+)</dc:identifier>", text):
                if "/handle/" in ident:
                    handles.append(ident.strip())
                    if len(handles) >= num_records:
                        break

            token_match = re.search(r"<resumptionToken[^>]*>([^<]+)</resumptionToken>", text)
            if token_match:
                resumption_token = token_match.group(1).strip()
                if not resumption_token:
                    break
            else:
                break

        browser.close()

    print(f"  Harvested {len(handles)} handle URLs")
    return handles[:num_records]


def _theseus_download(handles, pdf_dir: Path):
    """Download PDFs from handle URLs; skip already-downloaded files."""
    import time
    try:
        import requests
        from urllib.parse import urlparse
    except ImportError:
        print("  Missing dependency: requests\n  Run: pip install requests", file=sys.stderr)
        sys.exit(1)

    pdf_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for url in tqdm(handles, desc="downloading PDFs", unit="pdf"):
        handle_id = url.rstrip("/").split("/")[-1]
        dest = pdf_dir / f"theseus_{handle_id}.pdf"
        if dest.exists():
            downloaded.append(dest)
            continue
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            if "pdf" in resp.headers.get("Content-Type", ""):
                dest.write_bytes(resp.content)
            else:
                bitstream = _BitstreamFinder.find(resp.text)
                if not bitstream:
                    print(f"  No PDF found on page: {url}")
                    continue
                parsed = urlparse(url)
                pdf_url = f"{parsed.scheme}://{parsed.netloc}{bitstream}"
                pdf_resp = requests.get(pdf_url, stream=True, timeout=30)
                pdf_resp.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in pdf_resp.iter_content(8192):
                        f.write(chunk)
            downloaded.append(dest)
        except Exception as e:
            print(f"  Download error {url}: {e}")
        time.sleep(0.5)
    return downloaded


def _theseus_extract(pdf_files, data_dir: Path, dpi: int):
    """Extract page and paragraph crops from PDFs.

    For each page with >=2 text lines:
      1. One full-page image  (type="page")
      2. Up to 5 paragraph crops (type="paragraph"), each with >=2 text lines.
    """
    try:
        import fitz
        import pdfplumber
        import random as _random
        from PIL import Image as PILImage
    except ImportError as e:
        print(f"  Missing dependency: {e}\n  Run: pip install pymupdf pdfplumber pillow", file=sys.stderr)
        sys.exit(1)

    # Silence harmless MuPDF diagnostics (e.g. "No common ancestor in structure
    # tree") that flood stderr for Theseus PDFs with malformed tagged structures.
    fitz.TOOLS.mupdf_display_errors(False)
    fitz.TOOLS.mupdf_display_warnings(False)

    crops_dir = data_dir / "paragraph_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    records = []
    scale     = dpi / 72.0

    for pdf_path in tqdm(pdf_files, desc="extracting paragraphs", unit="pdf"):
        stem = pdf_path.stem
        crop_dir = crops_dir / stem
        crop_dir.mkdir(exist_ok=True)

        try:
            doc = fitz.open(str(pdf_path))
            page_imgs  = []
            for page in doc:
                pix = page.get_pixmap(dpi=dpi)
                page_imgs.append(PILImage.frombytes("RGB", (pix.width, pix.height), pix.samples))
            doc.close()

            with pdfplumber.open(str(pdf_path)) as pdf:
                for page_num, (page, pil_img) in enumerate(zip(pdf.pages, page_imgs)):
                    img_w, img_h = pil_img.size
                    text_lines   = page.extract_text_lines(layout=True)

                    if len(text_lines) < 2:
                        continue

                    # --- page-level record (full page image) ---
                    page_fname = f"p{page_num+1}_page.png"
                    pil_img.save(crop_dir / page_fname, dpi=(dpi, dpi))
                    page_text = "\n".join(ln["text"].strip() for ln in text_lines if ln["text"].strip())
                    records.append({
                        "source":    "theseus",
                        "file_name": f"{stem}_{page_fname}",
                        "pdf_file":  pdf_path.name,
                        "page":      page_num + 1,
                        "text":      page_text,
                        "type":      "page",
                        "image":     str(crop_dir / page_fname),
                    })

                    # --- paragraph-level records (up to 5) ---
                    line_bboxes = [(ln["x0"], ln["top"], ln["x1"], ln["bottom"]) for ln in text_lines]
                    min_top = min(b[1] for b in line_bboxes)
                    max_bottom = max(b[3] for b in line_bboxes)
                    content_height = max_bottom - min_top

                    if content_height <= 0:
                        continue

                    blocks = []
                    num_blocks = 5
                    rand_heights = [_random.randint(max(2, int(content_height // (num_blocks * 2))), int(content_height // num_blocks)) for _ in range(num_blocks)]
                    total_rand = sum(rand_heights)
                    if total_rand == 0:
                        continue
                    rand_heights = [h / total_rand * content_height for h in rand_heights]

                    cur_top = min_top
                    for bh in rand_heights:
                        block_top = cur_top
                        block_bottom = min(block_top + bh, max_bottom)
                        block_lines = [(x0, top, x1, bot) for x0, top, x1, bot in line_bboxes
                                      if top >= block_top and bot <= block_bottom]
                        if len(block_lines) < 2:
                            block_lines = [(x0, top, x1, bot) for x0, top, x1, bot in line_bboxes
                                           if top >= block_top and top < block_bottom]
                        if len(block_lines) < 2:
                            continue
                        blocks.append(block_lines)
                        cur_top = block_bottom

                    for pi, block_lines in enumerate(blocks[:5]):
                        pad = 2
                        left = min(b[0] for b in block_lines)
                        upper = min(b[1] for b in block_lines)
                        right = max(b[2] for b in block_lines)
                        lower = max(b[3] for b in block_lines)
                        left = int(max(0,     left  * scale - pad))
                        upper = int(max(0,     upper * scale - pad))
                        right = int(min(img_w, right * scale + pad))
                        lower = int(min(img_h, lower * scale + pad))
                        crop = pil_img.crop((left, upper, right, lower))
                        fname = f"p{page_num+1}_para{pi+1}.png"
                        crop.save(crop_dir / fname, dpi=(dpi, dpi))
                        block_top_min = min(b[1] for b in block_lines)
                        block_bot_max = max(b[3] for b in block_lines)
                        texts = [ln["text"].strip() for ln in text_lines
                                 if ln["top"] >= block_top_min and ln["bottom"] <= block_bot_max
                                 and ln["text"].strip()]
                        records.append({
                            "source":    "theseus",
                            "file_name": f"{stem}_{fname}",
                            "pdf_file":  pdf_path.name,
                            "page":      page_num + 1,
                            "text":      "\n".join(texts),
                            "type":      "paragraph",
                            "image":     str(crop_dir / fname),
                        })
        except Exception as e:
            print(f"  Error processing {pdf_path.name}: {e}")

    return records


def _theseus_build_parquet(records, data_dir: Path):
    """Write records to train.parquet with embedded PNG bytes."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        print(f"  Missing dependency: {e}", file=sys.stderr)
        sys.exit(1)

    sources, file_names, pdf_files, pages, texts, types, image_bytes_list = [], [], [], [], [], [], []
    for rec in tqdm(records, desc="building parquet", unit="row"):
        img_path = Path(rec["image"])
        if not img_path.exists():
            continue
        sources.append(rec["source"])
        file_names.append(rec["file_name"])
        pdf_files.append(rec["pdf_file"])
        pages.append(rec["page"])
        texts.append(rec["text"])
        types.append(rec.get("type", "paragraph"))
        image_bytes_list.append(img_path.read_bytes())

    table = pa.table({
        "source":    pa.array(sources,    pa.string()),
        "file_name": pa.array(file_names, pa.string()),
        "pdf_file":  pa.array(pdf_files,  pa.string()),
        "page":      pa.array(pages,      pa.int32()),
        "text":      pa.array(texts,      pa.string()),
        "type":      pa.array(types,      pa.string()),
        "image":     pa.array(
            [{"bytes": b, "path": n} for b, n in zip(image_bytes_list, file_names)],
            pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())]),
        ),
    })
    parquet = data_dir / "train.parquet"
    pq.write_table(table, parquet, compression="snappy")
    mb = parquet.stat().st_size / 1024 / 1024
    print(f"  Wrote {len(table)} rows → {parquet} ({mb:.1f} MB)")
    return parquet


def _theseus_name_func(row):
    """Build a unique image filename using the Theseus item ID.

    PDF names follow the pattern ``theseus_10024_xxxx_yy*.pdf`` where
    ``xxxx`` is the item identifier.  Images are named ``xxxx_{fname}``
    so that crops from different PDFs never collide.
    """
    pdf_name = row.get("pdf_file", "") or ""
    fname = row.get("file_name", "") or ""
    parts = Path(pdf_name).stem.split("_")
    # theseus_10024_xxxx_yy* → use parts[2] as the ID
    pdf_id = parts[2] if len(parts) >= 3 else Path(pdf_name).stem
    if pdf_id and fname and not fname.startswith(f"{pdf_id}_"):
        return f"{pdf_id}_{fname}"
    return fname


def cmd_theseus(args):
    pdf_dir  = Path(args.theseus_pdf_dir)
    data_dir = Path(args.theseus_data_dir)
    gen_dir  = Path(args.theseus_gen_dir)
    parquet  = data_dir / "train.parquet"
    data_dir.mkdir(parents=True, exist_ok=True)

    if parquet.exists():
        print(f"  [skip] {parquet} already exists (delete to re-build)")
    else:
        handles  = _theseus_harvest(THESEUS_OAI_URL, args.num_records)
        if not handles:
            print("  No handles harvested. Exiting.")
            return
        pdfs     = _theseus_download(handles, pdf_dir)
        if not pdfs:
            print("  No PDFs downloaded. Exiting.")
            return
        records  = _theseus_extract(pdfs, data_dir, args.dpi)
        print(f"  Extracted {len(records)} paragraph records from {len(pdfs)} PDFs")
        parquet  = _theseus_build_parquet(records, data_dir)

    if args.gen:
        _hf_gen(parquet, gen_dir, args.train_ratio, args.eval_ratio, args.kfolds,
                name_func=_theseus_name_func)


def cmd_theseus_pdfs(args):
    """Extract page and paragraph crops from already-downloaded PDFs in --theseus-pdf-dir."""
    pdf_dir  = Path(args.theseus_pdf_dir)
    data_dir = Path(args.theseus_data_dir)
    gen_dir  = Path(args.theseus_gen_dir)
    parquet  = data_dir / "train.parquet"

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"  No PDFs found in {pdf_dir}")
        return
    print(f"  Found {len(pdf_files)} PDFs in {pdf_dir}")

    if parquet.exists():
        print(f"  [skip] {parquet} already exists (delete to re-build)")
    else:
        data_dir.mkdir(parents=True, exist_ok=True)
        records  = _theseus_extract(pdf_files, data_dir, args.dpi)
        print(f"  Extracted {len(records)} records from {len(pdf_files)} PDFs")
        parquet  = _theseus_build_parquet(records, data_dir)

    if args.gen:
        _hf_gen(parquet, gen_dir, args.train_ratio, args.eval_ratio, args.kfolds,
                name_func=_theseus_name_func)


# ── subcommand: nlf ───────────────────────────────────────────────────────────

NLF_OUT_DIR   = "./data/nlf_ocr"
NLF_GEN_DIR   = "./data/nlf_ocr-gen"
NLF_GT_FI_DIR = "./data/nlf_ocr_groundtruth_fi"
NLF_GT_SV_DIR = "./data/nlf_ocr_groundtruth_sv"


def _nlf_parse_alto(xml_path):
    """Parse an ALTO XML ground-truth file.

    Returns ``(page_w, page_h, blocks)`` where each block is::

        {"bbox": (hpos, vpos, width, height), "lines": [str, ...]}

    Returns ``None`` if the page element or its dimensions are missing.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    # strip XML namespaces so we can use simple tag names
    for elem in root.iter():
        if isinstance(elem.tag, str) and "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]

    page = root.find(".//Page")
    if page is None:
        return None
    page_w = int(page.get("WIDTH", "0"))
    page_h = int(page.get("HEIGHT", "0"))
    if not page_w or not page_h:
        return None

    blocks = []
    for tb in root.findall(".//TextBlock"):
        hpos  = int(tb.get("HPOS", "0"))
        vpos  = int(tb.get("VPOS", "0"))
        width = int(tb.get("WIDTH", "0"))
        height = int(tb.get("HEIGHT", "0"))

        lines = []
        for tl in tb.findall(".//TextLine"):
            words = [s.get("CONTENT", "") for s in tl.findall(".//String")]
            line_text = " ".join(w for w in words if w).strip()
            if line_text:
                lines.append(line_text)

        if lines:
            blocks.append({"bbox": (hpos, vpos, width, height), "lines": lines})

    return page_w, page_h, blocks


def _nlf_detect_columns(blocks, page_w, page_h):
    """Detect text columns and assign each block to its column.

    Analyses the horizontal distribution of *blocks* (TextBlock bounding boxes
    from ALTO XML) to find vertical gutters.  Returns ``(num_cols, col_ids)``
    where ``col_ids[i]`` is the 0-based column index for *blocks*[i].

    Blocks wider than 65 % of the page are treated as full-width (column 0) —
    they are typically titles or headings that span all columns.

    Small / decorative blocks (page numbers, running headers) are excluded
    from gutter detection: a block must have ≥ 2 text lines **and** height
    ≥ 5 % of the page to influence column boundaries.  All blocks — including
    small ones — are still assigned to a column for reading-order purposes.
    """
    if not blocks or page_w <= 0:
        return 1, [0] * len(blocks)

    centres = []          # normalised x-centre of each block
    spans   = []          # normalised width  of each block
    for blk in blocks:
        hpos, vpos, w, h = blk["bbox"]
        centres.append((hpos + w / 2.0) / page_w)
        spans.append(w / page_w)

    # For gutter detection, only use significant blocks: ≥2 lines and tall
    # enough (filters out page numbers, folios, running headers).
    min_height = (page_h or 0) * 0.05
    narrow = sorted(
        c for c, s, blk in zip(centres, spans, blocks)
        if s < 0.65
        and len(blk["lines"]) >= 2
        and blk["bbox"][3] >= min_height
    )
    if len(narrow) < 2:
        return 1, [0] * len(blocks)

    # A gutter = gap between consecutive x-centres > 12 % of page width.
    threshold = 0.12
    bounds = []
    for j in range(1, len(narrow)):
        gap = narrow[j] - narrow[j - 1]
        if gap > threshold:
            bounds.append(narrow[j])

    num_cols = len(bounds) + 1

    # Assign ALL blocks (including small ones) to columns.
    col_ids = []
    for cx, sw in zip(centres, spans):
        if sw >= 0.65:                   # full-width → column 0
            col_ids.append(0)
            continue
        col = 0
        for b in bounds:
            if cx >= b:
                col += 1
            else:
                break
        col_ids.append(col)

    return num_cols, col_ids


NLF_PART_PREFIX = "train.part_"
NLF_NUM_PARTS   = 20
NLF_MAX_PARA_PER_COL = 10


def _nlf_merge_parts(out_dir, final_parquet):
    """Merge all ``train.part_*.parquet`` files into a single parquet.

    Uses fastparquet for reading (pyarrow 24 on Python 3.14 cannot read
    nested struct columns) and pyarrow for writing.  Deletes the part
    files after a successful merge.  Returns the total row count.
    """
    part_files = sorted(out_dir.glob(f"{NLF_PART_PREFIX}*.parquet"))
    if not part_files:
        return 0

    all_sources, all_file_names, all_texts, all_types, all_image_bytes = [], [], [], [], []

    for pf_path in part_files:
        import fastparquet as fp
        fpf = fp.ParquetFile(str(pf_path))
        df = fpf.to_pandas()

        all_sources.extend(df["source"].tolist()      if "source"    in df.columns else [])
        all_file_names.extend(df["file_name"].tolist() if "file_name" in df.columns else [])
        all_texts.extend(df["text"].tolist()           if "text"      in df.columns else [])
        all_types.extend(df["type"].tolist()           if "type"      in df.columns else [])

        if "image.bytes" in df.columns:
            all_image_bytes.extend(df["image.bytes"].tolist())
        elif "image" in df.columns:
            all_image_bytes.extend(
                r["bytes"] if isinstance(r, dict) else r
                for r in df["image"].tolist()
            )

    n = _write_parquet(all_sources, all_file_names, all_texts, all_image_bytes,
                       final_parquet, types=all_types)

    for pf in part_files:
        pf.unlink()

    return n


def cmd_nlf(args):
    """Extract page and paragraph crops from NLF OCR ground-truth (ALTO XML + TIFF).

    Processes ``nlf_ocr_groundtruth_fi`` and ``nlf_ocr_groundtruth_sv``
    directories. For each .tif / -gt2.xml pair:
      1. One full-page image   (type="page")
      2. One paragraph crop per TextBlock (type="paragraph"), all blocks

    Column layout is auto-detected per page (supports any number of columns,
    e.g. 1-col, 2-col, 4-col newspaper).  Reading order respects columns:
    column 0 (including full-width titles) first, then column 1, 2, 3…,
    top-to-bottom within each column.

    Records are written to up to 20 batch parquet files (``train.part_NN.parquet``),
    each covering ~5%% of the input files.  After all files are processed the
    parts are merged into ``train.parquet`` and deleted.  Pass ``--resume``
    to skip batches whose part file already exists.
    """
    try:
        import pyarrow as pa  # noqa: F401  (fail-fast probe)
        import pyarrow.parquet as pq  # noqa: F401
        from PIL import Image
    except ImportError as e:
        print(f"Missing dependency: {e}\nRun: pip install pyarrow pillow", file=sys.stderr)
        sys.exit(1)

    out_dir  = Path(args.nlf_dir)
    gen_dir  = Path(args.nlf_gen_dir) if args.nlf_gen_dir else out_dir.parent / f"{out_dir.name}-gen"
    parquet  = out_dir / "train.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine GT input directories.
    # Priority: --nlf-gt-dir (single dir) > --nlf-dir if it has .tif files
    #           > default fi/sv pair.
    if args.nlf_gt_dir:
        gt_dirs = [("nlf", args.nlf_gt_dir)]
    elif list(out_dir.glob("*.tif")):
        gt_dirs = [("nlf", str(out_dir))]
    else:
        gt_dirs = [
            ("nlf_fi", args.nlf_gt_fi_dir),
            ("nlf_sv", args.nlf_gt_sv_dir),
        ]

    if parquet.exists():
        print(f"  [skip] {parquet} already exists (delete to re-build)")
    else:
        import math

        # Clean up stale part files when NOT resuming.
        if not getattr(args, "resume", False):
            for stale in out_dir.glob(f"{NLF_PART_PREFIX}*.parquet"):
                stale.unlink()

        # Pre-count total TIF files across all gt_dirs.
        dir_tifs = []
        total_tifs = 0
        for source_label, gt_dir in gt_dirs:
            gt_path = Path(gt_dir)
            if not gt_path.is_dir():
                continue
            tifs = sorted(gt_path.glob("*.tif"))
            dir_tifs.append((source_label, gt_path, tifs))
            total_tifs += len(tifs)

        if total_tifs == 0:
            print("  ERROR: no .tif files found in any GT directory")
            return

        batch_size = max(1, math.ceil(total_tifs / NLF_NUM_PARTS))

        # Resume: count existing parts → skip that many files.
        existing_parts = sorted(out_dir.glob(f"{NLF_PART_PREFIX}*.parquet"))
        resume_batch = len(existing_parts) if getattr(args, "resume", False) else 0
        skip_files = resume_batch * batch_size
        if resume_batch:
            print(f"  [resume] {resume_batch} parts found, "
                  f"skipping first {skip_files} files")

        batch_idx = resume_batch
        files_in_batch = 0
        global_idx = 0
        b_sources, b_file_names, b_texts, b_types, b_image_bytes = [], [], [], [], []

        for source_label, gt_path, tif_files in dir_tifs:
            n_page, n_para = 0, 0
            col_counts = {}
            print(f"  {source_label}: {len(tif_files)} images in {gt_path}")

            for tif_path in tqdm(tif_files, desc=source_label, unit="img"):
                global_idx += 1

                # Resume: skip files belonging to already-written batches.
                if global_idx <= skip_files:
                    continue

                base = f"{source_label}__{tif_path.stem}"

                xml_path = gt_path / f"{tif_path.stem}-gt2.xml"
                if not xml_path.exists():
                    pass
                else:
                    parsed = _nlf_parse_alto(xml_path)
                    if parsed:
                        page_w, page_h, blocks = parsed
                        if blocks:
                            try:
                                img = Image.open(str(tif_path))
                                if img.mode != "RGB":
                                    img = img.convert("RGB")
                            except Exception as e:
                                print(f"  Error opening {tif_path.name}: {e}")
                                img = None

                            if img:
                                img_w, img_h = img.size
                                scale_x = img_w / page_w
                                scale_y = img_h / page_h

                                num_cols, col_ids = _nlf_detect_columns(blocks, page_w, page_h)
                                col_counts[num_cols] = col_counts.get(num_cols, 0) + 1

                                ordered = sorted(
                                    zip(blocks, col_ids),
                                    key=lambda bc: (bc[1], bc[0]["bbox"][1]),
                                )

                                # --- page-level record ---
                                page_lines = []
                                for blk, _ in ordered:
                                    page_lines.extend(blk["lines"])
                                page_text = "\n".join(page_lines)
                                if page_text.strip():
                                    buf = BytesIO()
                                    img.save(buf, format="PNG")
                                    b_sources.append(source_label)
                                    b_file_names.append(f"{base}__page.png")
                                    b_texts.append(page_text)
                                    b_types.append("page")
                                    b_image_bytes.append(buf.getvalue())
                                    n_page += 1

                                # --- paragraph-level records ---
                                col_groups = {}
                                for blk, col_id in ordered:
                                    bt = "\n".join(blk["lines"])
                                    if len(bt.strip()) >= 20:
                                        col_groups.setdefault(col_id, []).append(blk)

                                seen_texts = set()
                                pi = 0
                                for col_id in sorted(col_groups):
                                    col_blks = col_groups[col_id]
                                    n_blks = len(col_blks)
                                    col_count = 0
                                    for win in range(1, n_blks + 1):
                                        for start in range(0, n_blks - win + 1):
                                            window = col_blks[start:start + win]

                                            lines = []
                                            for b in window:
                                                lines.extend(b["lines"])
                                            combined = "\n".join(lines)

                                            if len(combined.strip()) < 20:
                                                continue
                                            if combined in seen_texts:
                                                continue
                                            seen_texts.add(combined)

                                            hpos_min = min(b["bbox"][0] for b in window)
                                            vpos_min = min(b["bbox"][1] for b in window)
                                            x1_max   = max(b["bbox"][0] + b["bbox"][2] for b in window)
                                            y1_max   = max(b["bbox"][1] + b["bbox"][3] for b in window)

                                            left   = max(0,    int(hpos_min * scale_x))
                                            upper  = max(0,    int(vpos_min * scale_y))
                                            right  = min(img_w, int(x1_max   * scale_x))
                                            lower  = min(img_h, int(y1_max   * scale_y))
                                            if right <= left or lower <= upper:
                                                continue

                                            crop = img.crop((left, upper, right, lower))
                                            buf = BytesIO()
                                            crop.save(buf, format="PNG")

                                            pi += 1
                                            col_count += 1
                                            b_sources.append(source_label)
                                            b_file_names.append(f"{base}__para{pi}.png")
                                            b_texts.append(combined)
                                            b_types.append("paragraph")
                                            b_image_bytes.append(buf.getvalue())
                                            n_para += 1

                                            if col_count >= NLF_MAX_PARA_PER_COL:
                                                break
                                        if col_count >= NLF_MAX_PARA_PER_COL:
                                            break

                files_in_batch += 1

                # Flush batch when it reaches batch_size or we're at the end.
                if files_in_batch >= batch_size or global_idx == total_tifs:
                    part_path = out_dir / f"{NLF_PART_PREFIX}{batch_idx:02d}.parquet"
                    n_ckpt = _write_parquet(b_sources, b_file_names, b_texts,
                                            b_image_bytes, part_path, types=b_types)
                    pct = global_idx / total_tifs
                    print(f"\n  [batch {batch_idx:02d}] {pct:.0%} — "
                          f"{global_idx}/{total_tifs} files, {n_ckpt} rows")
                    b_sources, b_file_names, b_texts, b_types, b_image_bytes = [], [], [], [], []
                    batch_idx += 1
                    files_in_batch = 0

            print(f"  {source_label}: {n_page} pages, {n_para} paragraphs")
            if col_counts:
                parts = ", ".join(f"{k}-col: {v}" for k, v in sorted(col_counts.items()))
                print(f"  {source_label} columns: {parts}")

        # Merge all parts into the final parquet.
        part_files = sorted(out_dir.glob(f"{NLF_PART_PREFIX}*.parquet"))
        n_parts = len(part_files)
        n = _nlf_merge_parts(out_dir, parquet)
        if n == 0:
            print("  ERROR: no valid image/xml pairs found")
            return
        mb = parquet.stat().st_size / 1024 / 1024
        print(f"\n  Merged {n_parts} parts → {n} rows → {parquet} ({mb:.1f} MB)")

    if args.gen:
        _hf_gen(parquet, gen_dir, args.train_ratio, args.eval_ratio, args.kfolds)


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
    p_hf.add_argument("--hf-mirror",   action="store_true", help="Use HF_ENDPOINT=https://hf-mirror.com")
    p_hf.add_argument("--gen",         action="store_true", help="Also generate train/eval/test JSONL in --hf-gen-dir")
    p_hf.add_argument("--train-ratio", type=float, default=float(os.environ.get("TRAIN_RATIO", "0.90")))
    p_hf.add_argument("--eval-ratio",  type=float, default=float(os.environ.get("EVAL_RATIO",  "0.05")))
    p_hf.add_argument("--kfolds",      type=int,   default=int(os.environ.get("KFOLDS", "0")),
                       help="K-fold cross-validation (0=disabled, >=2 generates fold_0/..fold_{k-1}/)")
    p_hf.set_defaults(func=cmd_hf)

    # digi-natlib
    p_dn = sub.add_parser("digi-natlib", parents=[shared], help="Download sdrobac/nodalida2017 and build data/digi-natlib/train.parquet")
    p_dn.add_argument("--dn-out-dir",     default=os.environ.get("DN_OUT_DIR",     DN_OUT_DIR))
    p_dn.add_argument("--dn-gen-dir",     default=os.environ.get("DN_GEN_DIR",     DN_GEN_DIR))
    p_dn.add_argument("--nodalida-clone", default=os.environ.get("NODALIDA_CLONE", NODALIDA_CLONE))
    p_dn.add_argument("--gen",            action="store_true", help="Also generate train/eval/test JSONL in --dn-gen-dir")
    p_dn.add_argument("--train-ratio",    type=float, default=float(os.environ.get("TRAIN_RATIO", "0.90")))
    p_dn.add_argument("--eval-ratio",     type=float, default=float(os.environ.get("EVAL_RATIO",  "0.05")))
    p_dn.add_argument("--kfolds",         type=int,   default=int(os.environ.get("KFOLDS", "0")),
                       help="K-fold cross-validation (0=disabled, >=2 generates fold_0/..fold_{k-1}/)")
    p_dn.set_defaults(func=cmd_digi_natlib)

    # synth
    p_synth = sub.add_parser("synth", parents=[shared], help="Generate synthetic Finnish fonts dataset")
    p_synth.add_argument("--synth-out-dir", default=os.environ.get("SYNTH_OUT_DIR", SYNTH_OUT_DIR))
    p_synth.add_argument("--samples", type=int,
                         default=int(os.environ.get("SYNTH_SAMPLES", str(SYNTH_SAMPLES_DEFAULT))),
                         help="Number of samples to generate")
    p_synth.add_argument("--single-line-ratio", type=float,
                         default=float(os.environ.get("SINGLE_LINE_RATIO", str(SYNTH_SINGLE_LINE_RATIO))),
                         help="Max fraction of single-line samples (default 0.01 = 1%%)")
    p_synth.add_argument("--train-ratio", type=float, default=0.90)
    p_synth.add_argument("--eval-ratio", type=float, default=0.05)
    p_synth.add_argument("--test-ratio", type=float, default=0.05)
    p_synth.add_argument("--hf-mirror", action="store_true", help="Use HF_ENDPOINT=https://hf-mirror.com")
    p_synth.set_defaults(func=cmd_synth)

    # theseus
    p_theseus = sub.add_parser("theseus", parents=[shared], help="Harvest, download, and extract Theseus PDFs")
    p_theseus.add_argument("--theseus-pdf-dir", default=os.environ.get("THESEUS_PDF_DIR", THESEUS_PDF_DIR))
    p_theseus.add_argument("--theseus-data-dir", default=os.environ.get("THESEUS_DATA_DIR", THESEUS_DATA_DIR))
    p_theseus.add_argument("--theseus-gen-dir", default=os.environ.get("THESEUS_GEN_DIR", THESEUS_GEN_DIR))
    p_theseus.add_argument("--dpi", type=int, default=int(os.environ.get("THESEUS_DPI", THESEUS_DPI)))
    p_theseus.add_argument("--num-records", type=int, default=int(os.environ.get("NUM_RECORDS", "10")))
    p_theseus.add_argument("--gen", action="store_true", help="Also generate train/eval/test JSONL")
    p_theseus.add_argument("--train-ratio", type=float, default=float(os.environ.get("TRAIN_RATIO", "0.90")))
    p_theseus.add_argument("--eval-ratio", type=float, default=float(os.environ.get("EVAL_RATIO", "0.05")))
    p_theseus.add_argument("--kfolds", type=int, default=int(os.environ.get("KFOLDS", "0")),
                           help="K-fold cross-validation (0=disabled, >=2 generates fold_0/..fold_{k-1}/)")
    p_theseus.set_defaults(func=cmd_theseus)

    # theseus-pdfs
    p_tpdfs = sub.add_parser("theseus-pdfs", parents=[shared], help="Extract from already-downloaded Theseus PDFs")
    p_tpdfs.add_argument("--theseus-pdf-dir", default=os.environ.get("THESEUS_PDF_DIR", THESEUS_PDF_DIR))
    p_tpdfs.add_argument("--theseus-data-dir", default=os.environ.get("THESEUS_DATA_DIR", THESEUS_DATA_DIR))
    p_tpdfs.add_argument("--theseus-gen-dir", default=os.environ.get("THESEUS_GEN_DIR", THESEUS_GEN_DIR))
    p_tpdfs.add_argument("--dpi", type=int, default=int(os.environ.get("THESEUS_DPI", THESEUS_DPI)))
    p_tpdfs.add_argument("--gen", action="store_true", help="Also generate train/eval/test JSONL")
    p_tpdfs.add_argument("--train-ratio", type=float, default=float(os.environ.get("TRAIN_RATIO", "0.90")))
    p_tpdfs.add_argument("--eval-ratio", type=float, default=float(os.environ.get("EVAL_RATIO", "0.05")))
    p_tpdfs.add_argument("--kfolds", type=int, default=int(os.environ.get("KFOLDS", "0")),
                         help="K-fold cross-validation (0=disabled)")
    p_tpdfs.set_defaults(func=cmd_theseus_pdfs)

    # nlf
    p_nlf = sub.add_parser("nlf", parents=[shared], help="Extract NLF OCR ground-truth (ALTO XML + TIFF)")
    p_nlf.add_argument("--nlf-dir", default=os.environ.get("NLF_OUT_DIR", NLF_OUT_DIR),
                       help="Output directory for train.parquet")
    p_nlf.add_argument("--nlf-gt-dir", default=None,
                       help="Single ground-truth directory (.tif + -gt2.xml). Overrides --nlf-gt-fi-dir/--nlf-gt-sv-dir")
    p_nlf.add_argument("--nlf-gt-fi-dir", default=os.environ.get("NLF_GT_FI_DIR", NLF_GT_FI_DIR),
                       help="Finnish ground-truth directory (.tif + -gt2.xml)")
    p_nlf.add_argument("--nlf-gt-sv-dir", default=os.environ.get("NLF_GT_SV_DIR", NLF_GT_SV_DIR),
                       help="Swedish ground-truth directory (.tif + -gt2.xml)")
    p_nlf.add_argument("--nlf-gen-dir", default=os.environ.get("NLF_GEN_DIR", None),
                       help="Output dir for JSONL (default: {nlf-dir}-gen)")
    p_nlf.add_argument("--gen", action="store_true", help="Also generate train/eval/test JSONL")
    p_nlf.add_argument("--train-ratio", type=float, default=float(os.environ.get("TRAIN_RATIO", "0.90")))
    p_nlf.add_argument("--eval-ratio", type=float, default=float(os.environ.get("EVAL_RATIO", "0.05")))
    p_nlf.add_argument("--kfolds", type=int, default=int(os.environ.get("KFOLDS", "0")),
                       help="K-fold cross-validation (0=disabled)")
    p_nlf.add_argument("--resume", action="store_true",
                       help="Resume from last checkpoint (train.checkpoint.parquet)")
    p_nlf.set_defaults(func=cmd_nlf)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
