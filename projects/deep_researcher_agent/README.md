# Deep Researcher Agent (MVP)
Minimal local Retrieval-Augmented-Generation style system:
- Upload PDF/TXT -> extract -> chunk -> embed -> FAISS index
- Query endpoint returns top chunks (with source)
How to run:
1. python3 -m venv venv && source venv/bin/activate
2. pip install -r requirements.txt
3. uvicorn src.api:app --reload --port 8000
Docs: http://localhost:8000/docs
