# Experiment 7 and 8 - Clinic FAQ RAG API (Groq + Hugging Face, deployable on Render)

A FastAPI service that answers questions from a small FAQ knowledge base.
Retrieval uses a small local embedding model (all-MiniLM-L6-v2 through fastembed/ONNX, no torch);
answers come from the Groq API. This keeps memory low enough for Render's free 512 MB instance.

## Files
- `app.py` - endpoints (`GET /`, `GET /health`, `POST /ask`), validation, logging, latency, per-IP rate limit
- `support_rag.py` - loading, chunking, embeddings, retrieval, Groq call
- `static/index.html` - web page served at `/`
- `data/` - knowledge base (.txt, .md, .pdf, .docx). Add your Experiment 5 files here.
- `test_api.py` - endpoint tests, CSV and two plots (`python test_api.py [base_url]`)
- `render.yaml` - Render blueprint, `.env.example` - configuration template

## Run locally with the real services
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then fill in GROQ_API_KEY
uvicorn app:app --reload
```
Open http://127.0.0.1:8000 (page) or http://127.0.0.1:8000/docs (API docs).

## Run offline, no keys (API wiring check only)
```bash
LLM_BACKEND=mock EMBEDDING_BACKEND=local uvicorn app:app
```
Answers are placeholders in this mode. It only proves the endpoints, validation and errors work.

## Tests
```bash
pip install -r requirements-dev.txt
python test_api.py                         # local
python test_api.py https://YOUR-APP.onrender.com   # deployed
```

## Notes
- `/health` shows the process is running and whether the index is built. It does not prove Groq or Hugging Face are reachable.
- The rate limit (`RATE_LIMIT_PER_MINUTE`, default 20 per IP) protects your free API quota. It is kept in memory and resets on restart.
- Not production-grade: no authentication, no persistent monitoring, no load testing.
- Educational content only, not medical advice.
