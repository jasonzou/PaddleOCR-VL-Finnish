"""
Render the first page of each PDF at 72, 144, 200, 300 DPI and save to test_dpi/.
Lets you visually compare quality and check file sizes across resolutions.
"""

import os
import glob
from pdf2image import convert_from_path

PDF_FOLDER = "theseus_pdfs"
OUT_FOLDER = "test_dpi"
RESOLUTIONS = [72, 144, 200, 300]


def main() -> None:
    pdf_files = sorted(glob.glob(os.path.join(PDF_FOLDER, "*.pdf")))
    if not pdf_files:
        print(f"No PDFs found in '{PDF_FOLDER}/'")
        return

    os.makedirs(OUT_FOLDER, exist_ok=True)

    for pdf_path in pdf_files:
        stem = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"\n{stem}")
        for dpi in RESOLUTIONS:
            pages = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=1)
            img = pages[0]
            out_path = os.path.join(OUT_FOLDER, f"{stem}_{dpi}dpi.png")
            img.save(out_path, format="PNG", dpi=(dpi, dpi))
            size_kb = os.path.getsize(out_path) / 1024
            print(f"  {dpi:>3} dpi  {img.size[0]:>5}×{img.size[1]:<5}  {size_kb:>7.1f} KB  → {out_path}")


if __name__ == "__main__":
    main()
