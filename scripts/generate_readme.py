#!/usr/bin/env python3
"""Generate README.md from dataset parquet files."""

import argparse
import pandas as pd
from pathlib import Path


def get_parquet_file(data_dir: Path):
    if (data_dir / "train-00001-of-00001.parquet").exists():
        return data_dir / "train-00001-of-00001.parquet"
    if (data_dir / "train.parquet").exists():
        return data_dir / "train.parquet"
    if (data_dir / "data_std.parquet").exists():
        return data_dir / "data_std.parquet"
    return data_dir / "data.parquet"


def get_size(data_dir: Path):
    if (data_dir / "train-00001-of-00001.parquet").exists():
        return len(pd.read_parquet(data_dir / "train-*-of-*.parquet"))
    if (data_dir / "train.parquet").exists():
        df = pd.read_parquet(data_dir / "train.parquet")
        return len(df)
    if (data_dir / "data_std.parquet").exists():
        df = pd.read_parquet(data_dir / "data_std.parquet")
        return len(df)
    df = pd.read_parquet(data_dir / "data.parquet")
    return len(df)


def generate_readme(data_dir: Path, name: str = None, description: str = None):
    name = name or data_dir.name.upper().replace("-", "_")
    name = name.replace(" ", "_")

    parquet_file = get_parquet_file(data_dir)
    df = pd.read_parquet(parquet_file)
    columns = df.columns.tolist()
    size = get_size(data_dir)

    if size < 1000:
        size_cat = "n<1K"
    elif size < 10000:
        size_cat = "1K<n<10K"
    elif size < 100000:
        size_cat = "10K<n<100K"
    else:
        size_cat = "100K<n<1M"

    features_yaml = []
    for col in columns:
        dtype = "string"
        if col == "image":
            dtype = "image"
        features_yaml.append(f"  - name: {col}")
        features_yaml.append(f"    dtype: {dtype}")

    features_block = "\n".join(features_yaml)

    configs_yaml = []
    configs_yaml.append("  - config_name: default")
    if (data_dir / "train-00001-of-00001.parquet").exists():
        configs_yaml.append("    data_files:")
        configs_yaml.append("    - split: train")
        configs_yaml.append("      path: train-*-of-*.parquet")
    elif (data_dir / "train.parquet").exists():
        configs_yaml.append("    data_files:")
        configs_yaml.append("    - split: train")
        configs_yaml.append("      path: train.parquet")
    else:
        configs_yaml.append("    data_files:")
        configs_yaml.append("    - split: train")
        configs_yaml.append("      path: data.parquet")
    configs_yaml.append("    default: true")

    configs_block = "\n".join(configs_yaml)

    fields_table = []
    fields_table.append("| Field | Type | Description |")
    fields_table.append("|-------|------|-------------|")
    for col in columns:
        if col == "file_name":
            desc = "Relative path to image file"
        elif col == "image":
            desc = "PNG image data"
        elif col == "text":
            desc = "Ground truth transcription"
        elif col == "id":
            desc = "Unique sample identifier"
        else:
            desc = ""
        dtype = "string" if col != "image" else "bytes"
        fields_table.append(f"| `{col}` | {dtype} | {desc} |")

    fields_block = "\n".join(fields_table)

    splits_table = []
    splits_table.append("| Split | Size |")
    splits_table.append("|-------|------|")
    splits_table.append(f"| train | {size:,} |")

    splits_block = "\n".join(splits_table)

    example_str = "{"
    for i, col in enumerate(columns):
        if col == "image":
            val = "<PNG bytes>"
        else:
            val = repr(df[col].iloc[0][:50])
        example_str += f'\n    "{col}": {val}'
        if i < len(columns) - 1:
            example_str += ","
    example_str += "\n}"

    readme = f"""---
license: cc-by-4.0
language:
- fi
pretty_name: {name}
size_categories:
- {size_cat}
task_categories:
- image-to-text
tags:
- OCR
- historic
configs:
{configs_block}
  features:
{features_block}
---

# {name}

## Quick Start

```python
from datasets import load_dataset

dataset = load_dataset("parquet", data_files="train.parquet", split="train")
```

## Dataset Description

{description or f"{name} is a Finnish OCR post-correction dataset containing {size:,} text pairs extracted from historic Finnish newspapers."}

### Supported Tasks

- **OCR Post-correction**: Correcting errors in OCR output

### Languages

Finnish (fi)

## Dataset Structure

### Data Fields

{fields_block}

### Data Splits

{splits_block}

### Example

```python
{example_str}
```

## Licensing

CC-BY 4.0
"""

    return readme


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Dataset directory")
    parser.add_argument("--name", help="Dataset name")
    parser.add_argument("--description", help="Dataset description")
    parser.add_argument("--output", help="Output README file (default: <dir>/README.md)")
    args = parser.parse_args()

    data_dir = Path(args.dir)
    readme = generate_readme(data_dir, args.name, args.description)

    output_path = Path(args.output) if args.output else data_dir / "README.md"
    output_path.write_text(readme)
    print(f"Created {output_path}")