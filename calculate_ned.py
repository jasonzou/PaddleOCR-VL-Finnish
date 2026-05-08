"""
Calculate NED, CER, and WER from model output JSONL files.

Each line must have:
  "answer" — model prediction
  "label"  — ground-truth text   (or fall back to messages[1].content)

Usage:
  python calculate_ned.py output/default
  python calculate_ned.py output/default --verbose
  python calculate_ned.py output/default --worst 20
  python calculate_ned.py output/default --by-source
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def levenshtein_distance(pred: str, ref: str) -> int:
    """Pure-Python Levenshtein distance (fallback)."""
    m, n = len(ref), len(pred)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == pred[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def ned(pred: str, ref: str) -> float:
    """Normalized edit distance in [0, 1]. 0 = perfect."""
    try:
        import Levenshtein
        dist = Levenshtein.distance(pred, ref)
    except ImportError:
        dist = levenshtein_distance(pred, ref)
    denom = max(len(pred), len(ref))
    return dist / denom if denom > 0 else 0.0


def cer(pred: str, ref: str) -> float:
    """Character Error Rate (CER) — Levenshtein distance / reference length."""
    try:
        import Levenshtein
        dist = Levenshtein.distance(pred, ref)
    except ImportError:
        dist = levenshtein_distance(pred, ref)
    denom = len(ref)
    return dist / denom if denom > 0 else 0.0


def wer(pred: str, ref: str) -> float:
    """Word Error Rate (WER) — Levenshtein distance on whitespace-split words."""
    pred_words = pred.split()
    ref_words = ref.split()
    try:
        import Levenshtein
        dist = Levenshtein.distance(pred_words, ref_words)
    except ImportError:
        # Convert to strings for Levenshtein
        pred_str = " ".join(pred_words)
        ref_str = " ".join(ref_words)
        dist = levenshtein_distance(pred_str, ref_str)
    denom = len(ref_words)
    return dist / denom if denom > 0 else 0.0


def load_results(path: Path) -> list[dict]:
    results = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(tqdm(f, desc=path.name, unit="line"), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  WARNING: {path}:{lineno}: JSON parse error: {e}", file=sys.stderr)
                continue
            pred = rec.get("answer", "")
            label = rec.get("label") or rec.get("messages", [{}] * 2)[1].get("content", "")
            results.append({
                "pred": pred,
                "label": label,
                "image": rec.get("images", [""])[0],
                "source": rec.get("source", path.stem),
            })
    return results


def evaluate(results: list[dict], by_source: bool = False) -> dict:
    if not results:
        return {"n": 0, "metrics": {}, "by_source": {}}

    neds = [ned(r["pred"], r["label"]) for r in tqdm(results, desc="computing NED", unit="sample", leave=False)]
    cers = [cer(r["pred"], r["label"]) for r in tqdm(results, desc="computing CER", unit="sample", leave=False)]
    wers = [wer(r["pred"], r["label"]) for r in tqdm(results, desc="computing WER", unit="sample", leave=False)]

    exact = sum(1 for r in results if r["pred"].strip() == r["label"].strip())
    total = len(results)

    def percentile(data: list[float], p: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = (p / 100) * (len(sorted_data) - 1)
        lo, hi = int(idx), min(int(idx) + 1, len(sorted_data) - 1)
        return sorted_data[lo] + (idx - lo) * (sorted_data[hi] - sorted_data[lo])

    metrics = {
        "ned": {
            "mean": sum(neds) / total,
            "std": (sum((s - sum(neds) / total) ** 2 for s in neds) / total) ** 0.5,
            "exact_match": exact / total,
            "p50": percentile(neds, 50),
            "p75": percentile(neds, 75),
            "p90": percentile(neds, 90),
            "p95": percentile(neds, 95),
            "p99": percentile(neds, 99),
        },
        "cer": {
            "mean": sum(cers) / total,
            "std": (sum((s - sum(cers) / total) ** 2 for s in cers) / total) ** 0.5,
            "p50": percentile(cers, 50),
            "p75": percentile(cers, 75),
            "p90": percentile(cers, 90),
            "p95": percentile(cers, 95),
            "p99": percentile(cers, 99),
        },
        "wer": {
            "mean": sum(wers) / total,
            "std": (sum((s - sum(wers) / total) ** 2 for s in wers) / total) ** 0.5,
            "p50": percentile(wers, 50),
            "p75": percentile(wers, 75),
            "p90": percentile(wers, 90),
            "p95": percentile(wers, 95),
            "p99": percentile(wers, 99),
        },
    }

    # NED buckets
    buckets = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.01]
    bucket_counts = [0] * (len(buckets) - 1)
    for s in neds:
        for i in range(len(buckets) - 1):
            if s < buckets[i + 1]:
                bucket_counts[i] += 1
                break
    metrics["ned"]["buckets"] = list(zip(buckets, buckets[1:], bucket_counts))

    # CER buckets
    cer_bucket_counts = [0] * (len(buckets) - 1)
    for s in cers:
        for i in range(len(buckets) - 1):
            if s < buckets[i + 1]:
                cer_bucket_counts[i] += 1
                break
    metrics["cer"]["buckets"] = list(zip(buckets, buckets[1:], cer_bucket_counts))

    # WER buckets (different range: WER can exceed 1.0)
    wer_buckets = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0, 1.5, 2.0, 5.0, 10.0]
    wer_bucket_counts = [0] * (len(wer_buckets) - 1)
    for s in wers:
        for i in range(len(wer_buckets) - 1):
            if s < wer_buckets[i + 1]:
                wer_bucket_counts[i] += 1
                break
    metrics["wer"]["buckets"] = list(zip(wer_buckets, wer_buckets[1:], wer_bucket_counts))

    result = {
        "n": total,
        "metrics": metrics,
        "by_source": {},
    }

    if by_source:
        by_src = defaultdict(list)
        for r in results:
            by_src[r["source"]].append(r)

        for src, src_results in by_src.items():
            src_n = len(src_results)
            src_neds = [ned(r["pred"], r["label"]) for r in src_results]
            src_cers = [cer(r["pred"], r["label"]) for r in src_results]
            src_wers = [wer(r["pred"], r["label"]) for r in src_results]
            src_exact = sum(1 for r in src_results if r["pred"].strip() == r["label"].strip())

            result["by_source"][src] = {
                "n": src_n,
                "ned": {"mean": sum(src_neds) / src_n, "exact_match": src_exact / src_n},
                "cer": {"mean": sum(src_cers) / src_n},
                "wer": {"mean": sum(src_wers) / src_n},
            }

    return result


def _bar(count: int, total: int, width: int = 20) -> str:
    filled = round(width * count / total) if total else 0
    return "█" * filled + "░" * (width - filled)


def print_report(name: str, data: dict, verbose: bool = False, worst_n: int = 0) -> None:
    n = data["n"]
    if n == 0:
        print(f"\n{name}: no results")
        return

    m = data["metrics"]

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Samples       : {n}")
    print(f"\n  NED (Normalized Edit Distance):")
    print(f"    Mean         : {m['ned']['mean']:.4f}  (lower is better)")
    print(f"    Std          : {m['ned']['std']:.4f}")
    print(f"    Exact match  : {m['ned']['exact_match']*100:.2f}%")
    print(f"    Percentiles   : p50={m['ned']['p50']:.4f}  p75={m['ned']['p75']:.4f}  p90={m['ned']['p90']:.4f}  p95={m['ned']['p95']:.4f}")

    print(f"\n  CER (Character Error Rate):")
    print(f"    Mean         : {m['cer']['mean']:.4f}")
    print(f"    Std          : {m['cer']['std']:.4f}")
    print(f"    Percentiles : p50={m['cer']['p50']:.4f}  p75={m['cer']['p75']:.4f}  p90={m['cer']['p90']:.4f}")

    print(f"\n  WER (Word Error Rate):")
    print(f"    Mean         : {m['wer']['mean']:.4f}")
    print(f"    Std          : {m['wer']['std']:.4f}")
    print(f"    Percentiles : p50={m['wer']['p50']:.4f}  p75={m['wer']['p75']:.4f}  p90={m['wer']['p90']:.4f}")

    print(f"\n  NED distribution:")
    for lo, hi, cnt in m["ned"]["buckets"]:
        label = f"  [{lo:.2f}, {'1.00' if hi > 1 else f'{hi:.2f}'})"
        pct = cnt / n * 100
        print(f"    {label:18s} {_bar(cnt, n)} {cnt:6d} ({pct:5.1f}%)")

    print(f"\n  CER distribution:")
    for lo, hi, cnt in m["cer"]["buckets"]:
        label = f"  [{lo:.2f}, {'1.00' if hi > 1 else f'{hi:.2f}'})"
        pct = cnt / n * 100
        print(f"    {label:18s} {_bar(cnt, n)} {cnt:6d} ({pct:5.1f}%)")

    print(f"\n  WER distribution:")
    for lo, hi, cnt in m["wer"]["buckets"]:
        if hi <= 1.0:
            label = f"  [{lo:.2f}, {hi:.2f})"
        elif hi <= 5.0:
            label = f"  [{lo:.2f}, {hi:.2f})"
        else:
            label = f"  [{lo:.1f}, {hi:.1f})"
        pct = cnt / n * 100
        print(f"    {label:18s} {_bar(cnt, n)} {cnt:6d} ({pct:5.1f}%)")

    if data.get("by_source"):
        print(f"\n  By source:")
        for src, src_data in sorted(data["by_source"].items()):
            print(f"    {src:<20} n={src_data['n']:5d}  "
                  f"NED={src_data['ned']['mean']:.4f}  "
                  f"CER={src_data['cer']['mean']:.4f}  "
                  f"WER={src_data['wer']['mean']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path, help="Output JSONL file(s) to evaluate")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-file details")
    parser.add_argument("--worst", "-w", type=int, default=0, metavar="N", help="Print N worst predictions")
    parser.add_argument("--by-source", "-s", action="store_true", help="Group metrics by source field")
    args = parser.parse_args()

    all_results = []
    for path in args.files:
        if not path.exists():
            print(f"ERROR: {path} not found", file=sys.stderr)
            continue
        results = load_results(path)
        data = evaluate(results, by_source=args.by_source)
        print_report(path.name, data, verbose=args.verbose, worst_n=args.worst)
        all_results.extend(results)

    if len(args.files) > 1 and all_results:
        combined = evaluate(all_results, by_source=args.by_source)
        print_report(f"COMBINED ({len(args.files)} files)", combined, worst_n=args.worst)


if __name__ == "__main__":
    main()