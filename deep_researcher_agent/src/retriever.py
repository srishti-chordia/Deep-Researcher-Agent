# src/retriever.py
import faiss
import numpy as np
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "index"
FAISS_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "metadata.json"

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def load_index():
    if not FAISS_PATH.exists():
        return None, None
    index = faiss.read_index(str(FAISS_PATH))
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def query(q: str, top_k: int = 5):
    index, metadata = load_index()
    if index is None:
        return {"error": "index not found. Build index first."}
    emb = EMBED_MODEL.encode(q).astype("float32")
    D, I = index.search(np.array([emb]), top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        meta = metadata[int(idx)]
        results.append(
            {
                "score": float(score),
                "source": meta["source"],
                "chunk_id": meta["chunk_id"],
                "text": meta["text"],
            }
        )
    return results
