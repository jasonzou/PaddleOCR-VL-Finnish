import os
import json
import requests
from html.parser import HTMLParser
from urllib.parse import urlparse, parse_qs
from sickle import Sickle
import pdfplumber
from tqdm import tqdm
import time


class BitstreamFinder(HTMLParser):
    """Parses a Theseus item page to locate the direct bitstream PDF link."""

    def __init__(self) -> None:
        super().__init__()
        self.bitstream_url: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del tag
        if self.bitstream_url:
            return
        attr_dict = dict(attrs)
        href = attr_dict.get("href") or ""
        path = href.split("?")[0]
        if "/bitstream/" in path and path.lower().endswith(".pdf"):
            self.bitstream_url = href.split("&")[0]  # drop &isAllowed=y


# ---------------------- CONFIGURATION ----------------------
# The OAI-PMH base URL for Theseus.
# This was identified from the repository's registry entry.
OAI_BASE_URL = "https://www.theseus.fi/oai/request"

# Folder to store downloaded PDFs
PDF_FOLDER = "theseus_pdfs"

# Folder to store the final JSON dataset
DATASET_FOLDER = "theseus_ocr_dataset"

# Number of PDFs to download for testing
NUM_PDFS_TO_DOWNLOAD = 10

# Image resolution for paragraph crops
DEFAULT_DPI = 300

# Metadata format to harvest (Dublin Core is standard)
METADATA_PREFIX = "oai_dc"

# ---------------------- STEP 1: HARVEST RECORDS ----------------------
def get_pdf_urls(num_records=NUM_PDFS_TO_DOWNLOAD):
    """
    Harvests metadata from Theseus OAI-PMH and extracts the direct PDF URLs.
    Theseus uses a handle system, so we need to construct the bitstream URL.
    """
    print(f"🌐 Connecting to OAI-PMH endpoint: {OAI_BASE_URL}")
    sickle = Sickle(OAI_BASE_URL)
    pdf_urls = []

    try:
        # `ListRecords` will iterate through all records, but we only need a few
        records = sickle.ListRecords(metadataPrefix=METADATA_PREFIX)
        print(f"📄 Harvesting metadata for {num_records} records...")

        for i, record in enumerate(records):
            if i >= num_records:
                break

            # Dublin Core metadata is in record.metadata
            metadata = record.metadata

            # The `dc:identifier` field often contains the handle URL.
            # Example handle: "https://www.theseus.fi/handle/10024/123412"
            identifiers = metadata.get("identifier", [])
            handle_url = None
            for identifier in identifiers:
                if "/handle/" in identifier:
                    handle_url = identifier
                    break

            if not handle_url:
                print(f"⚠️ No handle URL found in record {i+1}. Skipping.")
                continue

            print(f"✅ Found record {i+1}: {handle_url}")
            pdf_urls.append(handle_url)

            # Be polite to the server
            time.sleep(1)

    except Exception as e:
        print(f"❌ Error harvesting records: {e}")
        print("💡 Troubleshooting tips:")
        print("   - Ensure you have internet access and the OAI endpoint is reachable.")
        print("   - Try using the alternative base URL: http://publications.theseus.fi/oai/request")
        print("   - The server might be temporarily blocking your IP. Try again later.")

    return pdf_urls

# ---------------------- STEP 2: DOWNLOAD PDFS ----------------------
def download_pdfs(pdf_urls):
    """Downloads PDF files from the list of URLs."""
    os.makedirs(PDF_FOLDER, exist_ok=True)
    downloaded_files = []

    print(f"\n⬇️ Downloading {len(pdf_urls)} PDF files...")
    for i, url in enumerate(tqdm(pdf_urls, desc="Downloading")):
        try:
            # Extract filename from handle or generate a generic one
            parsed_url = urlparse(url)
            path_parts = parsed_url.path.split('/')
            handle_id = path_parts[-1] if path_parts[-1] else path_parts[-2]
            filename = f"theseus_{handle_id}.pdf"
            filepath = os.path.join(PDF_FOLDER, filename)

            # The handle URL returns an HTML page; extract the bitstream PDF link from it.
            html_response = requests.get(url, timeout=30)
            html_response.raise_for_status()

            content_type = html_response.headers.get("Content-Type", "")
            if "pdf" in content_type:
                pdf_response = html_response
            else:
                finder = BitstreamFinder()
                finder.feed(html_response.text)
                bitstream_links = [
                    line.strip() for line in html_response.text.splitlines()
                    if "/bitstream/" in line
                ]
                print(f"🔍 All bitstream hrefs in page ({len(bitstream_links)} lines):")
                for line in bitstream_links:
                    print(f"   {line}")
                print(f"🔍 Parsed bitstream_url: {finder.bitstream_url}")

                if not finder.bitstream_url:
                    print(f"❌ Could not find PDF bitstream link on page: {url}")
                    continue

                parsed = urlparse(url)
                bitstream_url = f"{parsed.scheme}://{parsed.netloc}{finder.bitstream_url}"
                pdf_response = requests.get(bitstream_url, stream=True, timeout=30)
                pdf_response.raise_for_status()

            with open(filepath, 'wb') as f:
                for chunk in pdf_response.iter_content(chunk_size=8192):
                    f.write(chunk)

            downloaded_files.append(filepath)
            print(f"   Saved: {filepath}")

        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to download {url}: {e}")
        except Exception as e:
            print(f"❌ An unexpected error occurred for {url}: {e}")

        # Be polite
        time.sleep(1)

    return downloaded_files

# ---------------------- STEP 3: EXTRACT TEXT ----------------------
def _render_pages_pymupdf(pdf_path, dpi: int):
    import fitz
    from PIL import Image
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    doc.close()
    return images


def _render_pages_pdf2image(pdf_path, dpi: int):
    from pdf2image import convert_from_path
    return convert_from_path(pdf_path, dpi=dpi)


def extract_text_from_pdf(pdf_path, dpi: int = DEFAULT_DPI, renderer: str = "pymupdf"):
    pdf_data = {
        "pdf_file": os.path.basename(pdf_path),
        "pages": []
    }

    pdf_stem = os.path.splitext(os.path.basename(pdf_path))[0]
    crop_dir = os.path.join(DATASET_FOLDER, "paragraph_crops", pdf_stem)
    os.makedirs(crop_dir, exist_ok=True)

    try:
        if renderer == "pymupdf":
            page_images = _render_pages_pymupdf(pdf_path, dpi=dpi)
        else:
            page_images = _render_pages_pdf2image(pdf_path, dpi=dpi)
        scale = dpi / 72.0

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, (page, pil_img) in enumerate(zip(pdf.pages, page_images)):
                text_lines = page.extract_text_lines(layout=True)
                lines = [line["text"] for line in text_lines if line["text"].strip()]

                para_groups: list[tuple[list[str], tuple[float, float, float, float]]] = []
                current_texts: list[str] = []
                current_bbox: list[float] = []
                last_bottom: float | None = None

                for line in text_lines:
                    line_text = line["text"].strip()
                    if not line_text:
                        continue
                    x0, top, x1, bottom = line["x0"], line["top"], line["x1"], line["bottom"]
                    if last_bottom is not None and (top - last_bottom) > 5:
                        if current_texts:
                            para_groups.append((current_texts, tuple(current_bbox)))  # type: ignore[arg-type]
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
                    para_groups.append((current_texts, tuple(current_bbox)))  # type: ignore[arg-type]

                img_w, img_h = pil_img.size
                paragraphs = []
                for para_idx, (texts, bbox) in enumerate(para_groups):
                    x0, top, x1, bottom = bbox
                    pad = 2
                    left  = max(0,     x0 * scale - pad)
                    upper = max(0,     top * scale - pad)
                    right = min(img_w, x1 * scale + pad)
                    lower = min(img_h, bottom * scale + pad)
                    crop = pil_img.crop((left, upper, right, lower))
                    crop_filename = f"p{page_num + 1}_para{para_idx + 1}.png"
                    crop.save(os.path.join(crop_dir, crop_filename), dpi=(dpi, dpi))
                    paragraphs.append({
                        "text": " ".join(texts),
                        "crop": os.path.join("paragraph_crops", pdf_stem, crop_filename),
                        "bbox": {"x0": x0, "top": top, "x1": x1, "bottom": bottom},
                    })

                page_info = {
                    "page_number": page_num + 1,
                    "text": page.extract_text(layout=True) or "",
                    "lines": lines,
                    "paragraphs": paragraphs,
                }
                pdf_data["pages"].append(page_info)

    except Exception as e:
        print(f"❌ Error processing {pdf_path}: {e}")

    return pdf_data

# ---------------------- MAIN EXECUTION ----------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Harvest and process Theseus.fi theses.")
    parser.add_argument(
        "--dpi", "-d",
        type=int,
        choices=[72, 144, 200, 400],
        default=DEFAULT_DPI,
        help=f"Crop image resolution in DPI (default: {DEFAULT_DPI})",
    )
    parser.add_argument(
        "--renderer",
        choices=["pymupdf", "pdf2image"],
        default="pymupdf",
        help="PDF renderer: pymupdf (no poppler) or pdf2image (requires poppler) (default: pymupdf)",
    )
    args = parser.parse_args()

    print("🚀 Starting OCR dataset creation from Theseus.fi\n")

    # Step 1: Get PDF URLs
    pdf_urls = get_pdf_urls()
    if not pdf_urls:
        print("❌ No PDF URLs found. Exiting.")
        return

    # Step 2: Download PDFs
    pdf_files = download_pdfs(pdf_urls)
    if not pdf_files:
        print("❌ No PDFs downloaded. Exiting.")
        return

    # Step 3: Extract text and build dataset
    print("\n📝 Extracting text from PDFs...")
    records = []
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_data = extract_text_from_pdf(pdf_file, dpi=args.dpi, renderer=args.renderer)
        for page in pdf_data["pages"]:
            for para in page["paragraphs"]:
                records.append({
                    "pdf_file": pdf_data["pdf_file"],
                    "page": page["page_number"],
                    "image": para["crop"],
                    "text": para["text"],
                })

    # Step 4: Save dataset as JSON
    os.makedirs(DATASET_FOLDER, exist_ok=True)
    dataset_path = os.path.join(DATASET_FOLDER, "theseus_ocr_dataset.json")
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Dataset created successfully!")
    print(f"📁 PDFs saved in: '{PDF_FOLDER}/'")
    print(f"📊 Dataset saved to: '{dataset_path}'")
    print(f"📄 Processed {len(pdf_files)} PDF files, {len(records)} paragraph records.")

if __name__ == "__main__":
    main()
