import argparse
import os
import json
import glob
import pdfplumber
from PIL import Image
from tqdm import tqdm

PDF_FOLDER = "theseus_pdfs"
DATASET_FOLDER = "theseus_ocr_dataset"


def _render_pages_pymupdf(pdf_path: str, dpi: int) -> list[Image.Image]:
    import fitz
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    doc.close()
    return images


def _render_pages_pdf2image(pdf_path: str, dpi: int) -> list[Image.Image]:
    from pdf2image import convert_from_path
    return convert_from_path(pdf_path, dpi=dpi)


def extract_text_from_pdf(pdf_path: str, resolution: int = 300,
                           renderer: str = "pymupdf") -> list[dict]:
    pdf_stem = os.path.splitext(os.path.basename(pdf_path))[0]
    crop_dir = os.path.join(DATASET_FOLDER, "paragraph_crops", pdf_stem)
    os.makedirs(crop_dir, exist_ok=True)

    records = []
    try:
        if renderer == "pymupdf":
            page_images = _render_pages_pymupdf(pdf_path, dpi=resolution)
        else:
            page_images = _render_pages_pdf2image(pdf_path, dpi=resolution)

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, (page, pil_img) in enumerate(zip(pdf.pages, page_images)):
                # pdfplumber coords are in PDF points; scale to pixels at target DPI.
                scale = resolution / 72.0

                text_lines = page.extract_text_lines(layout=True)

                # Group lines into paragraphs, tracking the bounding box of each.
                para_groups: list[tuple[list[str], tuple[float, float, float, float]]] = []
                current_texts: list[str] = []
                current_bbox: list[float] = []
                last_bottom: float | None = None

                for line in text_lines:
                    line_text = line["text"].strip()
                    if not line_text:
                        continue
                    x0, top, x1, bottom = line["x0"], line["top"], line["x1"], line["bottom"]
                    if last_bottom is not None and (top - last_bottom) > 12:
                        if current_texts:
                            para_groups.append((current_texts, (current_bbox[0], current_bbox[1], current_bbox[2], current_bbox[3])))
                            current_texts, current_bbox = [], []
                    if not current_texts:
                        current_bbox = [x0, top, x1, bottom]
                    else:
                        current_bbox[0] = min(current_bbox[0], x0)
                        current_bbox[1] = min(current_bbox[1], top)
                        current_bbox[2] = max(current_bbox[2], x1)
                        current_bbox[3] = max(current_bbox[3], bottom)
                    current_texts.append(line_text)
                    last_bottom = bottom

                if current_texts:
                    para_groups.append((current_texts, (current_bbox[0], current_bbox[1], current_bbox[2], current_bbox[3])))

                img_w, img_h = pil_img.size
                for para_idx, (texts, bbox) in enumerate(para_groups):
                    x0, top, x1, bottom = bbox
                    pad = 2
                    left   = max(0, x0 * scale - pad)
                    upper  = max(0, top * scale - pad)
                    right  = min(img_w, x1 * scale + pad)
                    lower  = min(img_h, bottom * scale + pad)
                    crop = pil_img.crop((left, upper, right, lower))
                    if crop.width < 28:
                        extra = (30 - crop.width + 1) // 2
                        crop = pil_img.crop((
                            max(0,     left  - extra),
                            max(0,     upper - extra),
                            min(img_w, right + extra),
                            min(img_h, lower + extra),
                        ))
                        print(f"  padded p{page_num + 1}_para{para_idx + 1}.png  ({crop.width}×{crop.height}px)  [{pdf_stem}]")
                    crop_filename = f"p{page_num + 1}_para{para_idx + 1}.png"
                    crop.save(os.path.join(crop_dir, crop_filename), dpi=(resolution, resolution))
                    records.append({
                        "pdf_file": os.path.basename(pdf_path),
                        "page": page_num + 1,
                        "image": os.path.join("paragraph_crops", pdf_stem, crop_filename),
                        "text": "\n".join(texts),
                    })

    except Exception as e:
        print(f"❌ Error processing {pdf_path}: {e}")

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract paragraph crops from PDFs.")
    parser.add_argument(
        "--resolution", "-r",
        type=int,
        choices=[72, 144, 200, 300],
        default=300,
        help="Crop image resolution in DPI (default: 300)",
    )
    parser.add_argument(
        "--renderer",
        choices=["pymupdf", "pdf2image"],
        default="pymupdf",
        help="PDF renderer: pymupdf (no poppler) or pdf2image (requires poppler) (default: pymupdf)",
    )
    args = parser.parse_args()

    pdf_files = sorted(glob.glob(os.path.join(PDF_FOLDER, "*.pdf")))
    if not pdf_files:
        print(f"No PDFs found in '{PDF_FOLDER}/'")
        return

    print(f"Processing {len(pdf_files)} PDFs at {args.resolution} dpi (renderer: {args.renderer})...")
    os.makedirs(DATASET_FOLDER, exist_ok=True)

    all_records = []
    for pdf_path in tqdm(pdf_files, desc="PDFs"):
        all_records.extend(extract_text_from_pdf(pdf_path, resolution=args.resolution, renderer=args.renderer))

    dataset_path = os.path.join(DATASET_FOLDER, "theseus_ocr_dataset.json")
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    print(f"✅ {len(all_records)} paragraph records saved to '{dataset_path}'")


if __name__ == "__main__":
    main()
