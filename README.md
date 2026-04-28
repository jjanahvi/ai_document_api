# 📄 AI Document Intelligence API

A production-ready REST API for PDF ingestion and conversational Q&A using **RAG**, **FastAPI**, **Streamlit**, and **Docker** — powered by Google Gemini.

![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![LangChain](https://img.shields.io/badge/LangChain-1.2-orange)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-purple)

---

## 🏗️ Architecture

```
┌─────────────────────┐        HTTP         ┌──────────────────────────┐
│   Streamlit UI      │ ──────────────────► │   FastAPI Backend        │
│   (port 8501)       │                     │   (port 8000)            │
│                     │ ◄────────────────── │                          │
│  Upload PDF         │    JSON responses   │  POST /ingest            │
│  Ask questions      │                     │  POST /query             │
│  View sources       │                     │  GET  /status            │
└─────────────────────┘                     │  DELETE /reset           │
                                            └──────────┬───────────────┘
                                                       │
                                            ┌──────────▼───────────────┐
                                            │   RAG Pipeline           │
                                            │  PyPDF → Chunker →       │
                                            │  Gemini Embeddings →     │
                                            │  ChromaDB (persisted)    │
                                            │      ↕                   │
                                            │  Similarity Search →     │
                                            │  Gemini LLM → Answer     │
                                            └──────────────────────────┘
```

## 📁 Project Structure

```
ai-document-api/
├── src/
│   ├── __init__.py
│   ├── schemas.py      # Pydantic v2 request/response models
│   ├── llm.py          # Gemini LLM initialisation
│   ├── rag.py          # Ingest + query pipeline
│   └── api.py          # FastAPI app with all endpoints
├── ui/
│   └── app.py          # Streamlit frontend
├── Dockerfile          # Container definition for the API
├── docker-compose.yml  # Runs API + UI together
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## 🚀 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Redirects to interactive docs |
| `GET` | `/status` | Health check + index statistics |
| `POST` | `/ingest` | Upload and index a PDF (multipart/form-data) |
| `POST` | `/query` | Ask a question against indexed docs |
| `DELETE` | `/reset` | Clear all indexed documents |

Interactive docs auto-generated at **http://localhost:8000/docs**

---

## 🐳 Running with Docker (recommended)

### Step 1 — Install Docker Desktop
Download from https://www.docker.com/products/docker-desktop
Install and start it. Verify:
```bash
docker --version
docker compose version
```

### Step 2 — Clone the repo
```bash
git clone https://github.com/jjanahvi/ai-document-api.git
cd ai-document-api
```

### Step 3 — Set your API key
```bash
cp .env.example .env
# Open .env and replace 'your_gemini_api_key_here' with your actual key
```

### Step 4 — Build and run
```bash
docker compose up --build
```
This builds the API container and starts both services. First run takes ~3 minutes to download and install dependencies.

### Step 5 — Open in browser
- **Streamlit UI**: http://localhost:8501
- **API docs**: http://localhost:8000/docs

### Step 6 — Stop
```bash
docker compose down
```

---

## 💻 Running Locally (without Docker)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API key
cp .env.example .env
# Edit .env with your key

# 3. Start FastAPI (terminal 1)
uvicorn src.api:app --reload --port 8000

# 4. Start Streamlit (terminal 2)
streamlit run ui/app.py

# 5. Open browser
# Streamlit: http://localhost:8501
# API docs:  http://localhost:8000/docs
```

---

## 🔑 Getting a Free Gemini API Key

1. Go to https://aistudio.google.com
2. Sign in with Google
3. Click **Get API Key** → Create API key
4. Free tier: 1,500 requests/day — no credit card needed

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| API framework | FastAPI | Async, auto-docs, Pydantic native |
| LLM | Google Gemini 2.0 Flash Lite | Free tier, fast |
| Embeddings | Gemini text-embedding-004 | Same API, no extra key |
| Vector store | ChromaDB | Lightweight, local, persistent |
| Orchestration | LangChain 1.2 | Standardised RAG interface |
| Data validation | Pydantic v2 | Request/response schemas |
| Frontend | Streamlit | Rapid UI, calls API over HTTP |
| Containerisation | Docker + Compose | Reproducible, deploy anywhere |

---

## 💡 Key Engineering Decisions

- **Lifespan context manager** — LLM loaded once at startup, not per-request
- **Deterministic doc IDs** — based on filename + content hash, avoids duplicates
- **Persistent ChromaDB volume** — documents survive container restarts
- **Proper HTTP status codes** — 201 Created, 404 Not Found, 413 Too Large, 500 errors
- **File size validation** — 20MB limit enforced before processing
- **Frontend calls API over HTTP** — clean separation, could swap frontend independently

---