# src/indexer.py
import faiss
import numpy as np
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "index"
VEC_PATH = INDEX_DIR / "vectors.npy"
META_PATH = INDEX_DIR / "metadata.json"
FAISS_PATH = INDEX_DIR / "faiss.index"


def build_faiss():
    if not VEC_PATH.exists():
        raise RuntimeError("Vectors file not found. Run embedder.build_index() first.")
    vectors = np.load(VEC_PATH)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    faiss.write_index(index, str(FAISS_PATH))
    print(f"FAISS index saved to {FAISS_PATH} (n={index.ntotal})")
