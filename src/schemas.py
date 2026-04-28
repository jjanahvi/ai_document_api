from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    """
    Response returned after successfully ingesting a document.

    Attributes:
        message: Human-readable confirmation message.
        doc_id: Deterministic identifier assigned to the ingested document.
        chunks_created: Number of text chunks stored in the vector store.
        filename: Original name of the uploaded file.
    """
    message: str
    doc_id: str
    chunks_created: int
    filename: str


class QueryRequest(BaseModel):
    """
    Request body for the /query endpoint.

    Attributes:
        question: The natural-language question to answer.
        k: Number of chunks to retrieve from the vector store.
           Defaults to 4. Higher values provide more context but
           increase latency and token usage.
    """
    question: str = Field(min_length=3, max_length=1000)
    k: int = Field(default=4, ge=1, le=10)


class SourceReference(BaseModel):
    """
    A single source chunk referenced in a query response.

    Attributes:
        page: 1-based page number from the source document.
        filename: Name of the source document file.
        chunk_index: Position of this chunk within its document.
    """
    page: int
    filename: str
    chunk_index: int


class QueryResponse(BaseModel):
    """
    Response returned by the /query endpoint.

    Attributes:
        answer: The LLM-generated answer grounded in retrieved chunks.
        sources: List of source chunks used to produce the answer.
        question: The original question echoed back for traceability.
    """
    answer: str
    sources: list[SourceReference]
    question: str


class StatusResponse(BaseModel):
    """
    Response returned by the /status endpoint.

    Attributes:
        status: Service health indicator, always 'ok' if reachable.
        documents_indexed: Number of documents currently in the vector store.
        chunks_indexed: Total number of chunks currently indexed.
        model: Name of the LLM model in use.
    """
    status: str
    documents_indexed: int
    chunks_indexed: int
    model: str