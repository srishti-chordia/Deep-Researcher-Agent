# src/embedder.py
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from src.chunker import chunk_text
from src.ingest import read_file_text


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
INDEX_DIR = ROOT / "data" / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"  # small & fast
MODEL = SentenceTransformer(MODEL_NAME)


def build_index():
    docs = []
    metadata = []
    vectors = []

    for f in RAW_DIR.iterdir():
        if not f.is_file():
            continue
        text = read_file_text(f)
        chunks = chunk_text(text)
        for i, ch in enumerate(chunks):
            docs.append({"source": str(f.name), "chunk_id": i, "text": ch})

    if not docs:
        print("No docs found in data/raw. Upload files first.")
        return

    texts = [d["text"] for d in docs]
    print("Encoding", len(texts), "chunks...")
    embeddings = MODEL.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # save vectors & metadata
    np.save(INDEX_DIR / "vectors.npy", embeddings.astype("float32"))
    with open(INDEX_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print("Saved vectors and metadata.")
