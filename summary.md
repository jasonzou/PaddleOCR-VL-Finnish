# Finnish OCR 
## Overview
**PaddleOCR-VL-Finnish** is a fine-tune model on **PaddleOCR-VL-1.5** (a Vision-Language OCR model) for Finnish language text recognition using **LoRA** fine-tuning, and **dpo** (Direct Preference Optimization).

---

## 1. Data Preparation

We try the following ways to prepare the data:

- Synthesis data
  - Search for open datasets
  - Search for Finnish OCR research papers, then identify datasets used in the papers and access the datasets
  - Evaluate datasets
  - Select datasets
  - Clean/Convert datasets

### 1.0 Dataset selection rationale

| Source | Rationale |
|--------|-----------|
| **AIDA (handwritten, typewritten, ship-info)** | Largest open-source Finnish OCR dataset from Kansallisarkisto (National Archives of Finland). Covers three distinct visual domains — handwritten historical documents, typewritten records, and ship-manifest tables — from a single source. The original dataset also contained synthetic data which was removed because we generate our own synthetic data with better typographic control. |
| **fin-13k / swe-11k** | Discovered through a literature review of Finnish OCR papers. fin-13k adds further real Finnish text-line diversity. swe-11k (Swedish) is included because Swedish is Finland's second official language, shares ä/ö diacritics, and the visual domain (historical newspapers) complements the Finnish datasets. |
| **Digi-Natlib (nodalida2017)** | From the Nodalida 2017 competition — widely referenced in Finnish OCR research papers. Contains digitised Finnish newspapers (digi, ~12k records) and National Library catalogue cards (natlib, ~54k records), each with aligned binarised images and ground-truth text. Two distinct visual styles (newsprint columns vs. library cards) in one source. |
| **Theseus** | Theseus.fi is the central thesis/dissertation repository for all Finnish universities of applied sciences, holding 340k+ PDFs. Chosen over other Finnish Language Bank sources because it supports OAI-PMH protocol for systematic harvesting and PDF page image rendering is straightforward. Provides academic Finnish prose at page and paragraph granularity — complementing the predominantly line-level datasets above. |
| **NLF OCR Ground Truth** | National Library of Finland OCR ground-truth dataset (~500k records), from digi.kansalliskirjasto.fi/opendata. Includes Finnish and Swedish documents with ALTO XML annotations (per-word bounding boxes) and high-resolution TIFF scans. Multi-column newspaper layouts (1-col through 4-col) provide essential layout-reading diversity that line-level datasets lack. |

More importantly, some extra mechnisms have been used to ensure the quality of datasets.
1. For each dataset, we went through randomly selection 100 records, and manually checked the selected records until all selected records have no issues.
2. For theseus, we used two OCR models (mineru and glm-ocr) to help. Only if the extracted content is matching to the outputs of mineru and glm-ocr, then the extracted content is included. Although in this way, some more difficult sitations are ignored, the quality of the dataset is garanteed.

### Procedures - Search for open datasets

I found https://huggingface.co/datasets/Kansallisarkisto/AIDA_ocr_training_data dataset. However, this dataset was created years ago. The dataset consists of synthesis data, typewritten, and handwritten data. In addition, the format of the data needs to be changed for this project. I removed the synthesis data and changed the format, then created three datasets on Huggingface:
- https://huggingface.co/datasets/caveman273/aida-handwritten
- https://huggingface.co/datasets/caveman273/aida-typewritten
- https://huggingface.co/datasets/caveman273/aida-ship-info

**Selection rationale:** AIDA is the largest open-source Finnish OCR dataset from the National Archives. Its three sub-components cover distinct visual domains (handwriting, typewriting, tabular data), providing complementary training signals.

**Cleanup method:**
- Original synthetic sub-component of AIDA removed (redundant with our own synth pipeline in 1.1).
- Schema normalised to `{text, image, file_name}` — column names are auto-detected among common variants (text/transcript/label/ground_truth, image/img/image_path, file_name/filename/id).
- Non-existent image paths and non-string text entries are skipped.
- All images normalised to PNG bytes for consistent serialisation.

### Procedures - Search for Finnish OCR research papers

Literature search for Finnish OCR research papers was carried out. From various papers, five datasets were identified. After cleaning up and formatting, I created the following on Huggingface:
- https://huggingface.co/datasets/caveman273/fin-13k
- https://huggingface.co/datasets/caveman273/swe-11k
The other two can be downloaded from 
- https://github.com/sdrobac/nodalida2017
- https://digi.kansalliskirjasto.fi/opendata/submit

**Selection rationale:** Academic papers on Finnish OCR cite these datasets as established benchmarks. fin-13k provides additional real Finnish text-line images. swe-11k covers Swedish (Finland's second official language, sharing ä/ö diacritics). The nodalida2017 dataset provides digitised Finnish newspapers and National Library catalogue cards. NLF ground truth (digi.kansalliskirjasto.fi) is the largest annotated Finnish OCR corpus available.

**Cleanup method:**
- fin-13k / swe-11k: Reformatted to HF datasets format with consistent {text, image, file_name} schema. Same auto-detected column conventions as AIDA.
- digi-natlib: Binary (.bin.png) / ground-truth (.gt.txt) pairs with empty text are skipped. Unique filenames constructed via `source__split__stem.png` to avoid cross-directory name collisions.
- NLF: Detailed cleanup below in section 1.6.

### Procedures - Other data sources
The Finnish Language bank was examined. The theseui.fi (theses and dissertation repository for Finnish Universities) was chosen as another data source because this site has over 340k PDF files, and it is relatively easier to convert PDF files to OCR datasets.

The site supports the OAI-PMH protocol. PDF files can be downloaded via OAI-PMH.

**Selection rationale:** Theseus provides academic Finnish prose — a text domain absent from the predominantly newspaper/library datasets above. With 340k+ PDFs available through a standard protocol (OAI-PMH), it offers scalable, reproducible data acquisition.

**Cleanup method:**
- Pages with < 2 text lines → skipped (insufficient content for paragraph extraction).
- Paragraph crops with < 2 text lines → skipped.
- Per-page paragraph extraction: full-page image saved (type=page), then up to 5 paragraph crops created by partitioning text-line regions (type=paragraph).
- PDFs rendered at 300 DPI for OCR-quality resolution.
- MuPDF structure-tree diagnostics suppressed (harmless for Theseus PDFs but noisy).
- Filename collisions prevented by prefixing with PDF-internal item ID. 

### Procedures - NLF OCR ground-truth 

NLF OCR ground-truth data undergoes the most thorough cleaning pipeline:

- **Unannotated scans**: TIF files without a corresponding `-gt2.xml` are silently skipped (unannotated scans exist in the dataset).
- **Page-level filtering**: ALTO XML is parsed for `<Page>`, `<TextBlock>`, and `<TextLine>` elements. Pages with zero text blocks or missing dimensions are skipped.
- **Column detection** (`_nlf_detect_columns`):
  - Blocks wider than 65% of the page width are treated as full-width (titles, headings) and assigned to column 0.
  - Small decorative blocks (page numbers, running headers) — defined as blocks with < 2 text lines AND height < 5% of the page — are excluded from column-boundary detection but still assigned to reading-order columns.
  - A "gutter" boundary between columns is detected when the gap between consecutive block x-centres exceeds 12% of the page width.
  - Supports any number of columns: 1-col, 2-col, 3-col, 4-col newspaper layouts.
- **Reading-order reconstruction**: Blocks are sorted by (column_id, then top-to-bottom y-position), ensuring correct text flow across multi-column pages.
- **Paragraph crop generation**: For each column, sliding windows of 1 to N consecutive text blocks are combined into paragraph crops. A minimum of 20 characters per paragraph is enforced. Duplicate paragraph texts (exact match) are de-duplicated per page. Up to 10 paragraph crops are extracted per column to prevent over-sampling visually repetitive regions.


## Datasets

| # | Dataset | Source | Type | Train size | Test size | Description |
|---|---------|--------|------|------------|-----------|-------------|
| 1 | aida-typewritten | caveman273/aida-typewritten (HF) | Text line | 31,300 | 4,272 | Typewritten business documents from the AIDA (Archive of Finnish Business Documents). Machine-typed legacy Finnish corporate/administrative text. |
| 2 | aida-handwritten | caveman273/aida-handwritten (HF) | Text line | 9,360 | 1,270 | Handwritten business documents from the AIDA archive. Cursive and hand-printed Finnish text on various paper types. |
| 3 | aida-ship-info | caveman273/aida-ship-info (HF) | Text line | 4,740 | 472 | Ship information records from the AIDA archive. Tabular/semi-structured maritime records with typed Finnish text. |
| 4 | fin-13k | caveman273/fin-13k (HF) | Text line | 13,000 | 653 | Finnish text image dataset. Diverse Finnish-language text lines extracted from scanned documents. |
| 5 | swe-11k | caveman273/swe-11k (HF) | Text line | 11,000 | 556 | Swedish text image dataset. Swedish-language text lines from digitized archives, useful for bilingual Finnish-Swedish OCR. |
| 6 | digi-natlib | sdrobac/nodalida2017 (GitHub) | Text line | 66,400 | 3,327 | Digitized Finnish newspapers (digi: 12.4k) and National Library catalogue cards (natlib: 54k) with ground truth. Published as part of Nodalida 2017. |
| 7 | theseus | theseus.fi (OAI-PMH) | Page / Paragraph | 44,500 | 2,230 | Finnish academic theses harvested via OAI-PMH from Theseus.fi. PDFs downloaded, rendered at 300 DPI; full-page images + paragraph crops (up to 5 per page) extracted via pdfplumber + PyMuPDF. |
| 8 | nlf-ocr | National Library of Finland | Page / Paragraph | 21,200 | 1,079 | NLF OCR ground-truth from ALTO XML + TIFF pairs. Finnish (`nlf_fi`) and Swedish (`nlf_sv`) collections. Full-page images + paragraph crops from ALTO TextBlock annotations with automatic multi-column layout detection. |
## Approach

Multi-dataset LoRA fine-tuning using Inverse Square Root Frequency sampling

### Inverse Square Root Frequency Sampling

Instead of naively repeating files, sampling weights are defined at the DataLoader level: P(dataset) ∝ 1/√N.

This formula is the gold standard for balancing "sufficient learning" with "preventing overfitting" (used in GPT multilingual training). It avoids the pitfalls of both:
- **1/N weighting**: inflates small-dataset weights too aggressively (→ overfitting)
- **Direct concatenation**: allows large datasets to drown out small ones

Precise normalized weights for the 8 real datasets (ready for config):

| Dataset | Samples | Weight (1/√N) | Normalized prob |
|---------|---------|---------------|-----------------|
| digi-natlib-gen  | 66,518 | 0.00388 | 6.0%  (raw 33.0%) |
| theseus-gen      | 44,591 | 0.00473 | 7.4%  (raw 22.1%) |
| aida_typewritten | 31,272 | 0.00566 | 8.8%  (raw 15.5%) |
| nlf-ocr          | 21,200 | 0.00687 | 10.7% (raw 10.5%) |
| aida_handwritten | 9,367  | 0.01033 | 16.1% (raw 4.6%)  |
| swe_11k          | 11,097 | 0.00950 | 14.8% (raw 5.5%)  |
| fin_13k          | 13,040 | 0.00876 | 13.6% (raw 6.5%)  |
| aida_ship_info   | 4,740  | 0.01453 | 22.6% (raw 2.4%)  |

**Result:** aida_ship_info exposure is boosted from a meager 2.4% to 22.6%. Because sampling is random with replacement, the model sees a re-sampled combination of its original 4.7k images each epoch — never the same physical repetition that would cause the model to memorize a few thousand images (overfitting).

## Results
### NED
| Dataset | Baseline | Lora | 
|---------|----------|------|-
| aida-typewritten  | 0.0530 | 0.0266 |
| aida-handwritten  | 0.3840 | 0.1412 |
| aida-ship-info    | 0.1538 | 0.0490 |
| fin-13k           | 0.1654 | 0.0606 |
| swe-11k           | 0.1389 | 0.0840 |
| digi-natlib       | 0.2201 | 0.1150 |
| theseus           | 0.6952 | 0.3408 | 
| nlf-ocr           | 0.9037 | 0.4804 |


### Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| LoRA rank/alpha | 8/8 | Conservative updates on noisy real documents |
| Learning Rate | 3e-4 | Balanced for multi-dataset fine-tuning |
| Optimiser | AdamW β1=0.9, β2=0.95 | Standard for transformer fine-tuning |
| Weight decay | 0.1 | Regularises LoRA matrices; prevents large adapter values |
| Warmup ratio | 0.1 | Stabilises early training; prevents gradient spikes |
| LR scheduler | Cosine, min_lr=5e-5 | Smooth decay; min_lr floor prevents drastic weight changes |
| Adamε | 1e-8 | Numerical stability for gradient updates |
| Mixed precision | bf16 | Matches base model precision; reduces memory |
| Seed | 23 | Reproducibility across runs |

### Image Augmentation

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

### Demo

- The model was released at Huggingface: https://huggingface.co/caveman273/ppocrvl-v1.5-finnish
- The demo site is hosted at Huggingface: https://caveman273-paddleocrvl-v1-5-finnish.hf.space/
- A Finnish OCR Evaluation dataset (14k) was released at Huggingface: https://huggingface.co/datasets/caveman273/Finnish-OCR-evaluation