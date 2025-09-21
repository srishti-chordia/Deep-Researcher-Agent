# src/api.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from src.ingest import save_uploadfile
from src.embedder import build_index
from src.indexer import build_faiss
from src.retriever import query as retrieve_query

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

app = FastAPI(title="Deep Researcher Agent (MVP)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    target = RAW_DIR / file.filename
    save_uploadfile(file, target)
    return {"status": "uploaded", "path": str(target)}


@app.post("/build_index")
def build_index_endpoint():
    # 1) create embeddings (calls embedder.build_index)
    build_index()
    # 2) build FAISS
    build_faiss()
    return {"status": "index_built"}


@app.post("/query")
def query_endpoint(q: str = Form(...), top_k: int = Form(5)):
    res = retrieve_query(q, top_k)
    return {"query": q, "results": res}


@app.get("/")
def root():
    return {"status": "running"}
