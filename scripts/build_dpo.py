#!/usr/bin/env python3
"""Build DPO dataset from base model evaluation results.

Examines results/base-* files (except base-synth-fonts), finds pairs where
the base model's prediction (answer) differs from the ground truth (label),
filters by NED/CER/WER thresholds, and generates synthetic rejected responses
using Finnish-specific OCR error patterns.

Generates dpo/train.jsonl with corresponding images in dpo/images/.
"""

import json
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

# Pull in NED/CER/WER from calculate_ned.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from calculate_ned import ned, cer, wer  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
DPO_DIR = PROJECT_ROOT / "dpo"
DPO_IMAGES_DIR = DPO_DIR / "images"
DPO_TRAIN_FILE = DPO_DIR / "train.jsonl"
DPO_EVAL_FILE = DPO_DIR / "eval.jsonl"
IMAGE_INDEX_FILE = DPO_DIR / ".image_index.json"

# Result file -> category suffix
RESULT_FILES = {
    "base-digi-natlib":   "digi-natlib",
    "base-hf-fin13k":     "fin13k",
    "base-hf-handwritten": "handwritten",
    "base-hf-shipinfo":   "shipinfo",
    "base-hf-swe11k":     "swe11k",
    "base-hf-typewritten": "typewritten",
    "base-theseus": "theseus",
    "base-nlfocr": "nlfocr",
}

IMAGE_SEARCH_DIRS = [
    "data/digi-natlib-gen/images",
    "data/hf_gen/images",
    "data/hf_gen/aida_handwritten/images",
    "data/hf_gen/aida_ship_info/images",
    "data/hf_gen/aida_typewritten/images",
    "data/hf_gen/fin_13k/images",
    "data/hf_gen/swe_11k/images",
    "data/nlf_ocr-gen/images",
    "data/theseus-gen/images",
]

# --- error metric thresholds ---
NED_MIN_THRESHOLD = 0.05
CER_THRESHOLD = 0.10

# --- Finnish OCR error transforms ---

FINNISH_DIACRITIC_SWAP = {
    "å": "a", "Å": "A",
    "ä": "a", "Ä": "A",
    "ö": "o", "Ö": "O",
    "é": "e", "É": "E",
    "ü": "u", "Ü": "U",
    "š": "s", "Š": "S",
    "ž": "z", "Ž": "Z",
}

COMMON_OCR_CONFUSIONS = [
    ("w", "v"), ("W", "V"),
    ("v", "w"), ("V", "W"),
    ("ö", "o"), ("Ö", "O"),
    ("ö", "0"),
    ("O", "0"),
    ("0", "O"),
    ("I", "l"),
    ("l", "I"),
    ("1", "l"),
    ("l", "1"),
    ("rn", "m"),
    ("m", "rn"),
    ("rn", "h"),
    ("cl", "d"),
    ("vv", "w"),
]

# Characters that commonly appear in Finnish OCR errors
VALID_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZäöåÄÖÅ0123456789"

# Punctuation that may appear in OCR output
VALID_PUNCT = ".,;:!?-\"'()[] /&%+=<>@#$€"

DIGITS = "0123456789"


def apply_diacritic_swap(text: str, prob: float = 0.6) -> str:
    result = []
    for ch in text:
        if ch in FINNISH_DIACRITIC_SWAP and random.random() < prob:
            result.append(FINNISH_DIACRITIC_SWAP[ch])
        else:
            result.append(ch)
    return "".join(result)


def apply_ocr_confusion(text: str, prob: float = 0.15) -> str:
    confusions = list(COMMON_OCR_CONFUSIONS)
    random.shuffle(confusions)
    applied = 0
    for src, dst in confusions:
        if random.random() < prob:
            if src in text:
                text = text.replace(src, dst, 1)
                applied += 1
                if applied >= 3:
                    break
    return text


def apply_random_noise(text: str, char_prob: float = 0.03) -> str:
    if not text:
        return text
    result = list(text)
    i = 0
    while i < len(result):
        r = random.random()
        if r < char_prob * 0.4:
            # Substitute with a visually similar or random Latin char
            result[i] = random.choice(VALID_CHARS)
        elif r < char_prob * 0.55:
            # Insert a random valid char
            result.insert(i, random.choice(VALID_CHARS))
        elif r < char_prob:
            # Delete
            del result[i]
            i -= 1
        i += 1
    return "".join(result)


def apply_case_noise(text: str, prob: float = 0.03) -> str:
    result = []
    for ch in text:
        if ch.isalpha() and random.random() < prob:
            result.append(ch.upper() if ch.islower() else ch.lower())
        else:
            result.append(ch)
    return "".join(result)


def apply_digit_swap(text: str, prob: float = 0.05) -> str:
    result = []
    for ch in text:
        if ch in DIGITS and random.random() < prob:
            result.append(random.choice(DIGITS))
        else:
            result.append(ch)
    return "".join(result)


def generate_synthetic_rejected(label: str, intensity: str = "medium") -> str:
    """Generate a plausible OCR-mistake version of the label.

    intensity: 'light' | 'medium' | 'heavy'
    """
    text = label

    if intensity == "light":
        diac_prob = random.uniform(0.2, 0.5)
        confusion_prob = random.uniform(0.03, 0.10)
        noise_prob = random.uniform(0.005, 0.02)
        case_prob = random.uniform(0.0, 0.02)
        digit_prob = random.uniform(0.0, 0.03)
    elif intensity == "heavy":
        diac_prob = random.uniform(0.5, 0.9)
        confusion_prob = random.uniform(0.10, 0.30)
        noise_prob = random.uniform(0.02, 0.08)
        case_prob = random.uniform(0.01, 0.06)
        digit_prob = random.uniform(0.04, 0.12)
    else:  # medium
        diac_prob = random.uniform(0.3, 0.7)
        confusion_prob = random.uniform(0.05, 0.20)
        noise_prob = random.uniform(0.01, 0.05)
        case_prob = random.uniform(0.0, 0.04)
        digit_prob = random.uniform(0.02, 0.08)

    # 1. Diacritic drops (core Finnish OCR error)
    text = apply_diacritic_swap(text, prob=diac_prob)

    # 2. Common OCR character confusions
    text = apply_ocr_confusion(text, prob=confusion_prob)

    # 3. Random char-level noise
    text = apply_random_noise(text, char_prob=noise_prob)

    # 4. Occasional case noise
    text = apply_case_noise(text, prob=case_prob)

    # 5. Digit swaps in numbers
    text = apply_digit_swap(text, prob=digit_prob)

    return text


def generate_synthetic_rejecteds(label: str, n: int = 5, intensity: str = "medium") -> list[str]:
    """Generate multiple synthetic rejected responses from a label."""
    synthetics = []
    for _ in range(n):
        synth = generate_synthetic_rejected(label, intensity=intensity)
        if synth and synth != label:
            synthetics.append(synth)
    return synthetics


def score_pair(answer: str, label: str) -> dict:
    return {
        "ned": ned(answer, label),
        "cer": cer(answer, label),
        "wer": wer(answer, label),
    }


def build_image_index():
    if IMAGE_INDEX_FILE.exists():
        with open(IMAGE_INDEX_FILE) as f:
            return json.load(f)

    print("Building image index (this may take a moment)...")
    index = {}
    for search_dir in IMAGE_SEARCH_DIRS:
        full_dir = PROJECT_ROOT / search_dir
        if not full_dir.is_dir():
            continue
        for f in full_dir.iterdir():
            if f.is_file() and not f.name.startswith("._"):
                if f.name not in index:
                    index[f.name] = search_dir

    with open(IMAGE_INDEX_FILE, "w") as f:
        json.dump(index, f)

    print(f"  Indexed {len(index)} unique images across {len(IMAGE_SEARCH_DIRS)} directories")
    return index


def build_dpo(max_samples: int = 0, train_ratio: float = 1.0, eval_ratio: float = 0.0,
              ned_min: float = NED_MIN_THRESHOLD, ned_max: float = 0.7):
    """Build DPO dataset.

    Args:
        max_samples: If > 0, limit total DPO pairs to this number.
        train_ratio: Fraction of pairs for train split.
        eval_ratio: Fraction of pairs for eval split.
        ned_min: Minimum NED score to include (exclude pairs below this).
        ned_max: Maximum NED score to include (exclude pairs above this).
    """
    DPO_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    image_index = build_image_index()

    all_entries = []
    used_images = set()
    total_real_pairs = 0
    total_synth_pairs = 0
    total_no_diff = 0
    total_missing = 0
    total_low_ned = 0
    total_dup_image = 0

    for result_name, category in RESULT_FILES.items():
        result_path = RESULTS_DIR / result_name
        if not result_path.exists():
            print(f"  SKIP: {result_path} not found")
            continue

        category_name = f"ocr-dpo-{category}"
        file_real = 0
        file_synth = 0
        file_total = 0
        file_no_diff = 0
        file_missing = 0
        file_low_ned = 0
        file_dup_image = 0

        with open(result_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                file_total += 1
                answer = entry.get("answer", "").strip()
                label = entry.get("label", "").strip()
                if answer == label:
                    file_no_diff += 1
                    continue

                scores = score_pair(answer, label)
                if scores["ned"] < ned_min:
                    file_low_ned += 1
                    continue
                if scores["ned"] > ned_max:
                    file_low_ned += 1
                    continue
                if scores["cer"] < CER_THRESHOLD:
                    file_low_ned += 1
                    continue

                img_rel = entry.get("images", [""])[0]
                img_filename = os.path.basename(img_rel)

                if img_filename not in image_index:
                    file_missing += 1
                    continue

                src_path = PROJECT_ROOT / image_index[img_filename] / img_filename
                if not src_path.exists():
                    file_missing += 1
                    continue

                if img_filename in used_images:
                    file_dup_image += 1
                    continue

                def make_entry(accepted: str, rejected: str) -> dict:
                    return {
                        "messages": [
                            {"role": "user", "content": "<image>OCR:"},
                            {"role": "assistant", "content": accepted},
                        ],
                        "chosen_response": [
                            {"role": "assistant", "content": accepted},
                        ],
                        "rejected_response": [
                            {"role": "assistant", "content": rejected},
                        ],
                        "images": [f"./images/{img_filename}"],
                        "category": category_name,
                        "_src_path": str(src_path),
                        "audios": [],
                        "videos": [],
                    }

                # Real model prediction as rejected
                real_entry = make_entry(label, answer)
                all_entries.append(real_entry)
                used_images.add(img_filename)
                file_real += 1

                # Generate synthetic rejected responses
                synthetics = generate_synthetic_rejecteds(label, n=3)
                for synth in synthetics:
                    if synth != answer and synth != label:
                        pass  # image already used by real entry above

        print(
            f"  {result_name}: {file_real} real + {file_synth} synth DPO pairs "
            f" (total={file_total}, no_diff={file_no_diff}, low_ned={file_low_ned}, missing={file_missing}, dup_img={file_dup_image})"
        )
        total_real_pairs += file_real
        total_synth_pairs += file_synth
        total_no_diff += file_no_diff
        total_missing += file_missing
        total_low_ned += file_low_ned
        total_dup_image += file_dup_image

    print(f"\n---")
    print(f"  Synth-only pairs from all categories:")

    # Also generate synthetic-only pairs from ALL items (including those where
    # the real model was already good), as long as images are available.
    synth_only_dup = 0
    for result_name, category in RESULT_FILES.items():
        result_path = RESULTS_DIR / result_name
        if not result_path.exists():
            continue

        category_name = f"ocr-dpo-{category}"
        file_synth_only = 0

        with open(result_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                label = entry.get("label", "").strip()
                if not label:
                    continue

                img_rel = entry.get("images", [""])[0]
                img_filename = os.path.basename(img_rel)
                if img_filename not in image_index:
                    continue

                if img_filename in used_images:
                    synth_only_dup += 1
                    continue

                src_path = PROJECT_ROOT / image_index[img_filename] / img_filename
                if not src_path.exists():
                    continue

                synthetics = generate_synthetic_rejecteds(label, n=2, intensity="light")
                seen = set()
                for synth in synthetics:
                    if synth != label and synth not in seen:
                        seen.add(synth)
                        if img_filename in used_images:
                            continue
                        all_entries.append({
                            "messages": [
                                {"role": "user", "content": "<image>OCR:"},
                                {"role": "assistant", "content": label},
                            ],
                            "chosen_response": [
                                {"role": "assistant", "content": label},
                            ],
                            "rejected_response": [
                                {"role": "assistant", "content": synth},
                            ],
                            "images": [f"./images/{img_filename}"],
                            "category": category_name,
                            "_src_path": str(src_path),
                            "audios": [], 
                            "videos": [],
                        })
                        used_images.add(img_filename)
                        file_synth_only += 1

        if file_synth_only > 0:
            print(f"    {result_name}: {file_synth_only} synth-only pairs")
            total_synth_pairs += file_synth_only

    total_dup_image += synth_only_dup

    # Shuffle to avoid ordering bias
    random.shuffle(all_entries)

    # Apply sample limit if specified (before caps)
    if max_samples > 0 and len(all_entries) > max_samples:
        all_entries = all_entries[:max_samples]
        print(f"\n  Limited to {max_samples} samples (--samples)")

    # Cap handwritten and shipinfo at 2.5% of total each (combined ≤5%)
    capped_categories = {"ocr-dpo-handwritten", "ocr-dpo-shipinfo"}
    capped_entries = [e for e in all_entries if e["category"] in capped_categories]
    if capped_entries and len(all_entries) > 0:
        max_per_cat = max(1, int(len(all_entries) * 0.025))
        cat_counts = {}
        kept = []
        for entry in all_entries:
            cat = entry["category"]
            if cat in capped_categories:
                count = cat_counts.get(cat, 0)
                if count < max_per_cat:
                    kept.append(entry)
                    cat_counts[cat] = count + 1
            else:
                kept.append(entry)
        removed = len(all_entries) - len(kept)
        if removed > 0:
            print(f"  Capped handwritten/shipinfo to 5% each (removed {removed} pairs)")
        all_entries = kept

    # Ensure theseus + nlfocr are at least 50% of total entries
    priority_categories = {"ocr-dpo-theseus", "ocr-dpo-nlfocr"}
    priority_entries = [e for e in all_entries if e["category"] in priority_categories]
    other_entries = [e for e in all_entries if e["category"] not in priority_categories]
    n_priority = len(priority_entries)
    n_other = len(other_entries)
    added = 0
    if n_priority < n_other and priority_entries:
        need = n_other - n_priority
        attempts = 0
        while added < need and attempts < need * 5:
            entry = random.choice(priority_entries)
            label = entry["chosen_response"][0]["content"]
            new_rejected = generate_synthetic_rejected(label)
            if new_rejected != label:
                clone = json.loads(json.dumps(entry))
                clone["rejected_response"] = [
                    {"role": "assistant", "content": new_rejected}
                ]
                clone["messages"][1]["content"] = label
                all_entries.append(clone)
                added += 1
            attempts += 1
        if added > 0:
            print(f"  Oversampled theseus/nlfocr: added {added} pairs to reach >=50%")

    DPO_TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)

    train_entries, eval_entries = all_entries, []
    if train_ratio > 0 and eval_ratio > 0:
        total_ratio = train_ratio + eval_ratio
        split = int(len(all_entries) * train_ratio / total_ratio)
        train_entries = all_entries[:split]
        eval_entries = all_entries[split:]

    # Copy images only for entries that made the final cut
    final_entries = train_entries + eval_entries
    seen_images = set()
    for entry in final_entries:
        img_filename = entry["images"][0].split("/")[-1]
        src_path = entry.pop("_src_path", None)
        if src_path and img_filename not in seen_images:
            seen_images.add(img_filename)
            dest_path = DPO_IMAGES_DIR / img_filename
            if not dest_path.exists():
                shutil.copy2(src_path, dest_path)
    print(f"  Copied {len(seen_images)} images to {DPO_IMAGES_DIR}")

    with open(DPO_TRAIN_FILE, "w", encoding="utf-8") as f:
        for entry in train_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if eval_entries:
        with open(DPO_EVAL_FILE, "w", encoding="utf-8") as f:
            for entry in eval_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"Done: {len(train_entries)} train + {len(eval_entries)} eval DPO pairs")
    print(f"  Train → {DPO_TRAIN_FILE}")
    if eval_entries:
        print(f"  Eval  → {DPO_EVAL_FILE}")
    print(f"  Real model predictions:   {total_real_pairs}")
    print(f"  Synthetic OCR errors:     {total_synth_pairs}")
    print(f"  No difference (skipped):  {total_no_diff}")
    print(f"  Excluded NED [{ned_min:.2f}, {ned_max:.2f}] / low CER: {total_low_ned}")
    print(f"  Missing images (skipped): {total_missing}")
    print(f"  Duplicate images (skipped): {total_dup_image}")
    print(f"\n  Train category distribution:")
    cats = Counter(e["category"] for e in train_entries)
    for cat, cnt in cats.most_common():
        pct = cnt / len(train_entries) * 100
        print(f"    {cat:<30s} {cnt:>6d}  ({pct:5.1f}%)")

    if eval_entries:
        print(f"\n  Eval category distribution:")
        eval_cats = Counter(e["category"] for e in eval_entries)
        for cat, cnt in eval_cats.most_common():
            pct = cnt / len(eval_entries) * 100
            print(f"    {cat:<30s} {cnt:>6d}  ({pct:5.1f}%)")

    print(f"\n  Unique images (train):")
    for cat in sorted(cats):
        imgs = set(e["images"][0].split("/")[-1] for e in train_entries if e["category"] == cat)
        print(f"    {cat:<30s} {len(imgs):>6d}")
    if eval_entries:
        print(f"\n  Unique images (eval):")
        for cat in sorted(cats):
            imgs = set(e["images"][0].split("/")[-1] for e in eval_entries if e["category"] == cat)
            print(f"    {cat:<30s} {len(imgs):>6d}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=0, help="Limit total DPO pairs (0 = no limit)")
    parser.add_argument("--train-ratio", type=float, default=1.0, help="Fraction for train split (default: 1.0)")
    parser.add_argument("--eval-ratio", type=float, default=0.0, help="Fraction for eval split (default: 0.0)")
    parser.add_argument("--ned-min", type=float, default=NED_MIN_THRESHOLD,
                       help=f"Minimum NED score to include (default: {NED_MIN_THRESHOLD})")
    parser.add_argument("--ned-max", type=float, default=0.7,
                       help="Maximum NED score to include (default: 0.7)")
    args = parser.parse_args()
    build_dpo(max_samples=args.samples, train_ratio=args.train_ratio, eval_ratio=args.eval_ratio,
              ned_min=args.ned_min, ned_max=args.ned_max)
