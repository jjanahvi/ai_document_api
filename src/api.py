"""
AI Document Intelligence API
FastAPI application exposing RAG pipeline as REST endpoints.

Endpoints:
    GET  /          — root, redirect to docs
    GET  /status    — service health + index statistics
    POST /ingest    — upload and index a PDF document
    POST /query     — ask a question against indexed documents
    DELETE /reset   — clear the vector store (dev/testing use)
"""
from dotenv import load_dotenv
load_dotenv()

import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import RedirectResponse

from src.llm import load_api_key, build_llm, MODEL_NAME
from src.rag import ingest_pdf, query_documents, get_store_stats, CHROMA_DIR
from src.schemas import (
    IngestResponse,
    QueryRequest,
    QueryResponse,
    StatusResponse,
)

# ── App state ─────────────────────────────────────────────────────────────────
# LLM and API key are initialised once at startup and reused across requests.
app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifespan — initialise shared resources on startup.

    The API key and LLM instance are loaded once at startup rather than
    per-request to avoid repeated environment lookups and model instantiation.
    """
    try:
        api_key = load_api_key()
        llm = build_llm()
        app_state["api_key"] = api_key
        app_state["llm"] = llm
        print(f"[startup] LLM ready: {MODEL_NAME}")
    except EnvironmentError as e:
        print(f"[startup] ERROR: {e}")
        raise

    yield  # Application runs here

    app_state.clear()
    print("[shutdown] Resources released.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Document Intelligence API",
    description=(
        "REST API for PDF ingestion and conversational question-answering "
        "using RAG (Retrieval-Augmented Generation) powered by Google Gemini."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to the interactive API documentation."""
    return RedirectResponse(url="/docs")


@app.get(
    "/status",
    response_model=StatusResponse,
    summary="Service health and index statistics",
    tags=["System"],
)
async def get_status() -> StatusResponse:
    """
    Return the current service health and vector store statistics.

    Provides a quick check that the API is reachable and shows how
    many documents and chunks are currently indexed.
    """
    api_key = app_state.get("api_key", "")
    docs, chunks = get_store_stats(api_key)

    return StatusResponse(
        status="ok",
        documents_indexed=docs,
        chunks_indexed=chunks,
        model=MODEL_NAME,
    )


@app.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a PDF document",
    tags=["Documents"],
)
async def ingest_document(file: UploadFile = File(...)) -> IngestResponse:
    """
    Upload a PDF file and index it into the vector store.

    The document is split into overlapping chunks, embedded using
    Google Gemini embeddings, and persisted to a local ChromaDB
    vector store. Each document receives a deterministic ID based
    on its filename and content hash.

    - **file**: A PDF file (multipart/form-data upload).

    Returns the document ID, number of chunks created, and filename.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are accepted. Please upload a .pdf file.",
        )

    file_bytes = await file.read()

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if len(file_bytes) > 20 * 1024 * 1024:  # 20MB limit
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds the 20MB size limit.",
        )

    try:
        doc_id, chunks_created = ingest_pdf(
            file_bytes=file_bytes,
            filename=file.filename,
            api_key=app_state["api_key"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}",
        )

    return IngestResponse(
        message="Document successfully ingested and indexed.",
        doc_id=doc_id,
        chunks_created=chunks_created,
        filename=file.filename,
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question against indexed documents",
    tags=["Documents"],
)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Answer a natural-language question using the indexed documents.

    Retrieves the most semantically similar chunks from the vector
    store and passes them as context to the LLM. The answer is
    grounded exclusively in the retrieved content.

    - **question**: Your question as a plain string.
    - **k**: Number of chunks to retrieve (default 4, max 10).

    Returns the answer, the source pages used, and the original question.
    """
    try:
        answer, sources = query_documents(
            question=request.question,
            api_key=app_state["api_key"],
            llm=app_state["llm"],
            k=request.k,
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}",
        )

    return QueryResponse(
        answer=answer,
        sources=sources,
        question=request.question,
    )


@app.delete(
    "/reset",
    summary="Clear the vector store",
    tags=["System"],
)
async def reset_store() -> dict:
    """
    Delete all indexed documents from the vector store.

    Intended for development and testing. Removes the entire
    ChromaDB persist directory and all indexed content.
    """
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        return {"message": "Vector store cleared successfully."}
    return {"message": "Vector store was already empty."}