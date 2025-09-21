# src/ingest.py
import os
from pathlib import Path
import pdfplumber

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

def save_uploadfile(uploadfile, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(uploadfile.file.read())
    return dest_path

def extract_text_from_pdf(path: Path) -> str:
    text_parts = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)

def read_file_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    else:
        # treat as plain text for .txt or unknown
        try:
            return path.read_text(encoding="utf-8")
        except:
            return path.read_text(encoding="latin-1")
