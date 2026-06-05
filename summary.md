# PaddleOCR-VL-Finnish — Project Summary

### Overview
**PaddleOCR-VL-Finnish** fine-tunes **PaddleOCR-VL-1.5** (a Vision-Language OCR model) for Finnish language text recognition using **LoRA** (rank-8) supervised fine-tuning, built for the **Paddle Global Model Derivative Competition**.

---

## 1. Data Preparation

We try the following ways to prepare the data:

- Synthesis data
  - Search for open datasets
  - Search for Finnish OCR research papers, then identify datasets used in the papers and access the datasets

### 1.1 Synthetic data

I identify popular Finnish fonts along with the top 10 Finnish datasets on Huggingface. Randomly selected Finnish fonts applying to the retrieved content from the top 10 Finnish datasets, a synthetic dataset is generated for the very first stage of fine-tuning. The purpose of this stage is to enhance the model with Finnish diacritics (å, ä, ö).

### 1.2 Search for open datasets

I found https://huggingface.co/datasets/Kansallisarkisto/AIDA_ocr_training_data dataset. However, this dataset was created years ago. The dataset consists of synthesis data, typewritten, and handwritten data. In addition, the format of the data needs to be changed for this project. I removed the synthesis data and changed the format, then created three datasets on Huggingface:
- https://huggingface.co/datasets/caveman273/aida-handwritten
- https://huggingface.co/datasets/caveman273/aida-typewritten
- https://huggingface.co/datasets/caveman273/aida-ship-info

### 1.3 Search for Finnish OCR research papers

Literature search for Finnish OCR research papers was carried out. From various papers, five datasets were identified. After cleaning up and formatting, I created the following on Huggingface:
- https://huggingface.co/datasets/caveman273/fin-13k
- https://huggingface.co/datasets/caveman273/swe-11k
The other two can be downloaded from 
- https://github.com/sdrobac/nodalida2017
- https://digi.kansalliskirjasto.fi/opendata/submit

The build_ocr_dataset.py can generate suitable dataset files (compatible with Huggingface datasets).

### 1.4 Other data sources
The Finnish Language bank was examined. The theseui.fi (theses and dissertation repository for Finnish Universities) was chosen as another data source because this site has over 340k PDF files, and it is relatively easier to convert PDF files to OCR datasets.

The site supports the OAI-PMH protocol. PDF files can be downloaded via OAI-PMH. 

### 1.5 Summary

Each dataset has been formatted as { "text", "image", "file_name"}.


| dataset  | Descriptions | 
|---------- |----------------|
| Synth-Font Scale | 50 k records 
| typewritten | 31.3 k records |
| handwritten | 9.36 k records |
| ship-info | 4.74 k records |
| fin-13k | 13 k records |
| swe-11k | 11 k records |
| theseus | 50 k records |
| digi-data | 12.4 k records |
| natlib-data | 54 k records |
| nlf ocr groud truth | 500 k records |

## 2. Fine-tuning

A **6-stage curriculum** with LoRA on PaddleOCR-VL-1.5 using PaddleFormers framework. Finnish is natively supported (111 languages) — no tokeniser modifications needed.

### 2.1 Framework Configuration

| Aspect | Value |
|--------|-------|
| Sequence length | 16384 |
| Padding-free | True (NaViT dynamic resolution) |
| LoRA rank | 8 (default; varies per stage: 16 → 8 → 4) |
| Optimiser | AdamW β1=0.9, β2=0.95, wd=0.1 |
| Mixed precision | bf16 |
| Evaluation metric | NED (Normalised Edit Distance) |
| Template | `paddleocr_vl_v15` + custom plugin |

### 2.2 Multi-Stage Curriculum

| Stage | Dataset(s) | Prob | Epochs | Learning Rate | Resume From |
|-------|-----------|------|--------|--------------|-------------|
| 0 – Synthetic Warm-up | synth-fonts (50k) | 1.0 | 3 | 1e-4 | PaddleOCR-VL-1.5 |
| 1 – Foundation | hf (57k) | 1.0 | 5 | 5e-4 | stage0_synth |
| 2 – Digi-Natlib | digi-natlib + hf | 0.8, 0.2 | 3 | 2.5e-4 | stage1_hf |
| 3 – NLF | nlf + hf + digi-natlib | 0.5, 0.25, 0.25 | 3 | 1e-4 | stage2_digi |
| 4 – Theseus Pages | theseus + hf + digi-natlib + nlf | 0.7, 0.1, 0.1, 0.1 | 3 | 5e-5 | stage3_nlf |
| 5 – Balanced Mix | hf + digi-natlib + nlf + theseus | 0.25 each | 4 | 5e-5 | stage4_theseus |

**Probability rationale:**
- **Stage 0–1**: Single dataset, no replay needed (pure foundation/synthetic)
- **Stage 2**: 80/20 split — prioritise new domain (digi-natlib) while replaying hf to prevent catastrophic forgetting
- **Stage 3**: 50/25/25 — nlf is a large new domain (328k), gets majority weight; equal replay of prior domains maintains their performance
- **Stage 4**: 70/10/10/10 — theseus is a new domain (pages/blocks vs textlines), heavy focus with small replay to retain previous domain knowledge
- **Stage 5**: 25/25/25/25 — final balanced mix to harmonise all domains at equal weight

### 2.3 Target NED Values

| Domain | Target NED |
|--------|-----------|
| hf | < 0.05 |
| digi-natlib | < 0.08 |
| nlf | < 0.10 |
| theseus | < 0.12 |

### 2.5 Fine-Tune Parameters

| Stage | LoRA rank | LoRA alpha | Learning Rate | Epochs | Rationale |
|-------|-----------|------------|--------------|--------|-----------|
| 0 – Synthetic Warm-up | 16 | 16 | 1e-4 | 3 | High rank for rapid Finnish diacritic (å, ä, ö) adaptation on clean synthetic data |
| 1 – Foundation | 8 | 16 | 5e-4 | 5 | Halve rank after warm-up; higher LR for quick domain adaptation |
| 2 – Digi-Natlib | 8 | 8 | 2.5e-4 | 3 | Reduced alpha/rank ratio (1:1) for conservative updates on noisy real documents |
| 3 – NLF | 8 | 8 | 1e-4 | 3 | Stable config; lower LR as adapter already captures Finnish patterns |
| 4 – Theseus Pages | 4 | 8 | 5e-5 | 3 | Minimal rank for small domain; lowest LR to preserve prior knowledge |
| 5 – Balanced Mix | 4 | 8 | 5e-5 | 4 | Final harmonisation; equal mix needs gentle updates |

**Shared training parameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimiser | AdamW β1=0.9, β2=0.95 | Standard for transformer fine-tuning |
| Weight decay | 0.1 | Regularises LoRA matrices; prevents large adapter values |
| Warmup ratio | 0.1 | Stabilises early training; prevents gradient spikes |
| LR scheduler | Cosine, min_lr=5e-5 | Smooth decay; min_lr floor prevents drastic weight changes |
| Adamε | 1e-8 | Numerical stability for gradient updates |
| Mixed precision | bf16 | Matches base model precision; reduces memory |
| Seed | 23 | Reproducibility across runs |

**Why reduce rank and LR across stages?**
- Early stages need higher capacity to learn Finnish grapheme shapes and textline geometry
- Later stages focus on adapting to new domains (digi-natlib, nlf, theseus) — less capacity needed as the adapter already captures Finnish patterns
- Lower rank and LR in later stages reduces catastrophic forgetting of earlier domain knowledge

### 3. Training Data Construction Rigor 

- **Collection process**: Sources traced to HuggingFace (`caveman273/aida-*`) and Nodalida 2017 competition (GitHub `sdrobac/nodalida2017`); reproducible scripts (`build_ocr_dataset.py`)
- **Annotation completeness**: Schema documented (`file_name`, `text`, `image`); pipeline produces parquet → JSONL with embedded images
- **Quality control**: Dataset checked via `check_dataset.py`; evaluation metrics (NED, CER, WER) used to validate model quality
- **Data analysis**: Statistics generated by `check_dataset.py`; README reports base vs. LoRA performance breakdown

### 4. Image Augmentation in the Fine-Tuning Pipeline

PaddleOCR-VL-1.5 uses a **NaViT-style dynamic-resolution vision encoder** — no manual image resizing is needed. The model automatically resizes images to optimal multiple-of-patch-size dimensions via `get_smarted_resize`.

**Augmentation architecture** (`PaddleOCRVLV15Plugin` in `vl_v15_template.py`):

Two modes — **classic** (per-entry `p` gate) and **policy-driven** (two-layer sampling):

1. **Policy gate**: decides *how many* and *which categories* of transforms to apply
2. **Eligibility gate**: each entry's own `p` field determines if it's a candidate

**Category weights** (policy mode): region_mask: 0.15, noise: 0.25, distortion: 0.20, degradation: 0.20, geometric: 0.10, padding: 0.10. The proportions can be adjusted in `data/aug_config.yaml`.

**Augmentation classes available:**

| Category | Transforms |
|----------|-----------|
| **Noise (whole-image)** | `GaussianNoise`, `JpegCompression`, `AlbumentationsISONoise` |
| **Blur (whole-image)** | `GaussianBlur`, `AlbumentationsMotionBlur` |
| **Colour** | `ColorJitter` (brightness/contrast/saturation/hue) |
| **Distortion** | `Curve` (sine-wave), `AlbumentationsGridDistortion`, `AlbumentationsOpticalDistortion` |
| **Dropout** | `DropoutHorizontal`, `DropoutVertical`, `DropoutRand` |
| **Region masks** | `MaskRandomBlock`, `MaskRandomPixel`, `MaskGaussianNoise`, `MaskSaltPepper`, `MaskBlur`, `MaskMosaic` |
| **Geometric** | `ImagePadding`, `LineOverlay`, `TextBorder` |

**Sequence length filtering**: With max_seq_len=16384, extremely wide images can generate excessive vision tokens. A character limit of **1200 characters** per sample is applied to prevent token overflow. Samples exceeding this limit are discarded (ground truth is never truncated). 

### 5. Preliminary Demo

- The model was released at Huggingface: https://huggingface.co/caveman273/ppocrvl-v1.5-finnish
- The demo site is hosted at Huggingface: https://caveman273-paddleocrvl-v1-5-finnish.hf.space/
