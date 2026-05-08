"""
Randomly sample records from a dataset (parquet or JSONL) and display text + images, or save to a directory.

Usage:
  python check_dataset.py                          # uses data/digi-natlib/train.parquet
  python check_dataset.py data/digi-natlib-gen/test.jsonl
  python check_dataset.py --summary                # dataset statistics only
  python check_dataset.py --save                   # save images to output-dir
  python check_dataset.py --save --output-dir extracts
"""

import argparse
import json
import math
import random
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path


def load_data(path):
    """Load data from parquet or JSONL file."""
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path), "parquet"
    elif path.suffix == ".jsonl":
        samples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples, "jsonl"
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def get_record(data, data_type, idx):
    """Extract a record from data based on type."""
    if data_type == "parquet":
        row = data.slice(idx, 1).to_pydict()
        return {
            "source": row["source"][0],
            "file_name": row["file_name"][0],
            "text": row["text"][0],
            "image": row["image"][0],
        }
    else:
        rec = data[idx]
        return {
            "source": rec.get("source", "unknown"),
            "file_name": rec.get("images", [""])[0] if rec.get("images") else "",
            "text": rec.get("messages", [{}])[1].get("content", ""),
            "image": rec.get("images", [""])[0] if rec.get("images") else "",
        }


def get_text(data, data_type, idx):
    """Extract text from a record."""
    if data_type == "parquet":
        return data[idx]["text"]
    else:
        msgs = data[idx].get("messages", [])
        return msgs[1].get("content", "") if len(msgs) > 1 else ""


def get_image(data, data_type, idx, data_path=None):
    """Extract image bytes from a record."""
    if data_type == "parquet":
        img_struct = data[idx]["image"]
        raw = img_struct["bytes"] if isinstance(img_struct, dict) else img_struct
        return raw
    else:
        img_path = data[idx].get("images", [""])[0]
        if not img_path:
            return None
        if data_path:
            base_dir = Path(data_path).parent
            resolved = base_dir / img_path.lstrip("./")
            if resolved.exists():
                return resolved.read_bytes()
        elif Path(img_path).exists():
            return Path(img_path).read_bytes()
        return None


def cmd_summary(data, data_type, path):
    try:
        from PIL import Image
    except ImportError as e:
        print(f"Missing dependency: {e}\nRun: pip install pillow", file=sys.stderr)
        sys.exit(1)

    n = len(data)
    print(f"  Total rows: {n:,}")

    if data_type == "jsonl":
        texts = [get_text(data, data_type, i) for i in range(n)]
        sources = [get_record(data, data_type, i).get("source", "unknown") for i in range(n)]
    else:
        rows = data.to_pydict()
        texts = rows["text"]
        sources = rows["source"]

    char_lens = [len(t) for t in texts if t]
    word_lens = [len(t.split()) for t in texts if t]
    total_chars = sum(char_lens)
    total_words = sum(word_lens)

    sample_size = min(2000, n)
    sample_idxs = random.sample(range(n), sample_size)
    widths, heights, img_bytes_sizes = [], [], []
    for i in sample_idxs:
        img_data = get_image(data, data_type, i)
        if img_data:
            img_bytes_sizes.append(len(img_data))
            img = Image.open(BytesIO(img_data))
            widths.append(img.size[0])
            heights.append(img.size[1])

    def pct(lst, p):
        s = sorted(lst)
        return s[int(len(s) * p / 100)]

    source_counts = Counter(sources)

    if data_type == "parquet":
        total_img_mb = sum(
            len((data[i]["image"]["bytes"] if isinstance(data[i]["image"], dict) else data[i]["image"]))
            for i in range(n)
        ) / 1024 / 1024
        print(f"  Parquet file size : {path.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"  Embedded image data : {total_img_mb:.1f} MB")

    print()
    print(f"  Source breakdown:")
    for src, cnt in sorted(source_counts.items()):
        print(f"    {src:<20} {cnt:>8,}  ({cnt/n*100:.1f}%)")
    print()

    print(f"  Text (characters):")
    print(f"    total             : {total_chars:,}")
    print(f"    mean              : {total_chars/n:.1f}")
    print(f"    min / p25 / median / p75 / max  : "
          f"{min(char_lens)} / {pct(char_lens,25)} / {pct(char_lens,50)} / {pct(char_lens,75)} / {max(char_lens)}")
    print()

    print(f"  Text (words):")
    print(f"    total             : {total_words:,}")
    print(f"    mean              : {total_words/n:.1f}")
    print(f"    min / p25 / median / p75 / max  : "
          f"{min(word_lens)} / {pct(word_lens,25)} / {pct(word_lens,50)} / {pct(word_lens,75)} / {max(word_lens)}")
    print()

    if widths:
        print(f"  Image dimensions (sampled {sample_size:,} rows):")
        print(f"    width  — mean: {sum(widths)/len(widths):.0f}  "
              f"min/p25/median/p75/max: {min(widths)}/{pct(widths,25)}/{pct(widths,50)}/{pct(widths,75)}/{max(widths)}")
        print(f"    height — mean: {sum(heights)/len(heights):.0f}  "
              f"min/p25/median/p75/max: {min(heights)}/{pct(heights,25)}/{pct(heights,50)}/{pct(heights,75)}/{max(heights)}")
        print(f"    bytes  — mean: {sum(img_bytes_sizes)/len(img_bytes_sizes):.0f}  "
              f"min/median/max: {min(img_bytes_sizes)}/{pct(img_bytes_sizes,50)}/{max(img_bytes_sizes)}")


def cmd_sample(data, data_type, data_path, args):
    try:
        from PIL import Image
    except ImportError as e:
        print(f"Missing dependency: {e}\nRun: pip install pillow", file=sys.stderr)
        sys.exit(1)

    n_total = len(data)
    rng = random.Random(args.seed)
    idxs = sorted(rng.sample(range(n_total), min(args.n, n_total)))
    print(f"Sampling {len(idxs)} records: rows {idxs}\n")

    if args.save:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, idx in enumerate(idxs):
            rec = get_record(data, data_type, idx)
            source = rec["source"]
            file_name = rec["file_name"]
            text = rec["text"]

            img_data = get_image(data, data_type, idx, data_path)
            if img_data:
                img = Image.open(BytesIO(img_data))
            else:
                print(f"[{idx:05d}] No image found, skipping")
                continue

            if file_name:
                base_name = Path(file_name).stem
                ext = Path(file_name).suffix
            else:
                base_name = f"{idx:05d}"
                ext = ".jpg"
            if not ext or ext not in (".jpg", ".jpeg", ".png", ".gif", ".bmp"):
                ext = ".jpg"

            out_path = output_dir / f"{base_name}{ext}"
            img.save(out_path)

            text_preview = text[:50] + "..." if len(text) > 50 else text
            print(f"[{idx:05d}] {out_path.name} — {img.size[0]}×{img.size[1]} — {text_preview}")

        print(f"\nSaved {len(idxs)} images to {output_dir.absolute()}")
    else:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError as e:
            print(f"Missing dependency: {e}\nRun: pip install matplotlib", file=sys.stderr)
            sys.exit(1)

        n_cols = 2
        n_rows = (len(idxs) + n_cols - 1) // n_cols
        fig = plt.figure(figsize=(14, 4 * n_rows))
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.5, wspace=0.3)

        for plot_i, idx in enumerate(idxs):
            rec = get_record(data, data_type, idx)
            source = rec["source"]
            file_name = rec["file_name"]
            text = rec["text"]

            img_data = get_image(data, data_type, idx, data_path)
            if img_data:
                img = Image.open(BytesIO(img_data))
            else:
                print(f"[{idx:05d}] No image found, skipping")
                continue

            print(f"── record {plot_i} ──────────────────────────────────")
            print(f"  source   : {source}")
            print(f"  file_name: {file_name}")
            print(f"  text     : {text}")
            print(f"  image    : {img.size[0]}×{img.size[1]} {img.mode}")
            print()

            ax = fig.add_subplot(gs[plot_i // n_cols, plot_i % n_cols])
            ax.imshow(img, cmap="gray" if img.mode in ("L", "1") else None, aspect="auto")
            ax.set_title(f"[{plot_i}] source={source}\nfile={file_name}\n{text}", fontsize=7, loc="left", wrap=True)
            ax.axis("off")

        plt.suptitle(f"{data_path.name} — {len(idxs)} random records", fontsize=10)
        plt.show()


def cmd_gen_ratio(data, data_type, data_path, args):
    """Generate train/eval/test JSONL splits from a parquet file."""
    if data_type != "parquet":
        print("ERROR: --gen-ratio only works with parquet files", file=sys.stderr)
        sys.exit(1)

    rows = data.to_pydict()
    n = len(rows["text"])
    print(f"  Generating splits with train_ratio={args.train_ratio}, eval_ratio={args.eval_ratio}")

    entries = []
    for i in range(n):
        text = rows["text"][i]
        if not isinstance(text, str) or not text.strip():
            continue
        entries.append({
            "messages": [
                {"role": "user", "content": "<image>OCR:"},
                {"role": "assistant", "content": text},
            ],
            "images": [f"./images/{rows['file_name'][i]}"],
            "source": rows["source"][i],
        })

    print(f"  Total valid entries: {len(entries)}")

    rng = random.Random(args.seed)
    rng.shuffle(entries)

    train_end = int(len(entries) * args.train_ratio)
    eval_end = train_end + int(len(entries) * args.eval_ratio)

    splits = {
        "train": entries[:train_end],
        "eval": entries[train_end:eval_end],
        "test": entries[eval_end:],
    }

    for split_name, rows_list in splits.items():
        out_path = data_path.parent / f"{data_path.stem}_{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows_list:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {split_name}: {len(rows_list)} entries → {out_path}")


def cmd_gen_ds_ratio(paths, args):
    """Compute dataset sampling probabilities for multi-dataset training."""
    sizes = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            sizes[str(p)] = sum(1 for line in f if line.strip())

    print("Dataset sizes:")
    for path, size in sorted(sizes.items(), key=lambda x: -x[1]):
        print(f"  {Path(path).name}: {size:,}")

    method = args.ratio_method
    total = sum(sizes.values())

    if method == "equal":
        probs = {n: 1.0 / len(sizes) for n in sizes}
    elif method == "sqrt":
        sqrt_sizes = {n: math.sqrt(s) for n, s in sizes.items()}
        sqrt_total = sum(sqrt_sizes.values())
        probs = {n: sqrt_sizes[n] / sqrt_total for n in sizes}
    else:  # proportional
        probs = {n: s / total for n, s in sizes.items()}

    print(f"\n{method} probabilities:")
    for name in sorted(probs.keys()):
        print(f"  {Path(name).name}: {probs[name]:.4f} ({probs[name] * 100:.1f}%)")

    if args.yaml:
        print("\n# YAML config:")
        print("train_dataset_path:")
        for p in paths:
            print(f"  - {p}")
        prob_strs = [f"{probs[str(p)]:.4f}" for p in paths]
        print(f"train_dataset_prob: \"{', '.join(prob_strs)}\"")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gen-ds-ratio", action="store_true", help="Compute dataset ratios for multi-dataset training")
    parser.add_argument("--ratio-method", default="sqrt", choices=["proportional", "equal", "sqrt"],
                        help="Ratio computation method (default: sqrt)")
    parser.add_argument("--yaml", action="store_true", help="Output YAML config for --gen-ds-ratio")
    parser.add_argument("files", nargs="*", type=Path, help="JSONL files (or single file for other modes)")
    parser.add_argument("--n", type=int, default=10, help="Number of records to sample (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--summary", action="store_true", help="Print dataset statistics only")
    parser.add_argument("--save", action="store_true", help="Save images to output-dir instead of displaying")
    parser.add_argument("--output-dir", default="samples", help="Output directory for --save (default: samples)")
    parser.add_argument("--gen-ratio", action="store_true", help="Generate train/eval/test JSONL splits from parquet")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio (default: 0.8)")
    parser.add_argument("--eval-ratio", type=float, default=0.1, help="Eval split ratio (default: 0.1)")
    args = parser.parse_args()

    if args.gen_ds_ratio:
        if not args.files:
            print("ERROR: --gen-ds-ratio requires JSONL files", file=sys.stderr)
            sys.exit(1)
        cmd_gen_ds_ratio(args.files, args)
    else:
        if not args.files:
            print("ERROR: no data file specified", file=sys.stderr)
            sys.exit(1)
        
        data_path = args.files[0]
        if not data_path.exists():
            print(f"ERROR: {data_path} not found", file=sys.stderr)
            sys.exit(1)

        data, data_type = load_data(data_path)
        n_total = len(data)
        print(f"Dataset: {data_path} ({n_total:,} rows) [{data_type}]")

        if args.gen_ratio:
            cmd_gen_ratio(data, data_type, data_path, args)
        elif args.summary:
            cmd_summary(data, data_type, data_path)
        else:
            cmd_sample(data, data_type, data_path, args)


if __name__ == "__main__":
    main()