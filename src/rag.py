import hashlib
import os
import tempfile
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from src.schemas import SourceReference

CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
EMBED_MODEL = "gemini-embedding-001"


def build_embeddings(api_key: str) -> GoogleGenerativeAIEmbeddings:
    """
    Instantiate the embedding model.

    Args:
        api_key: Google API key.

    Returns:
        A GoogleGenerativeAIEmbeddings instance.
    """
    return GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
    )


def compute_doc_id(filename: str, content: bytes) -> str:
    """
    Compute a deterministic document ID from filename and content hash.

    Args:
        filename: Original filename of the uploaded document.
        content: Raw bytes of the document.

    Returns:
        A hex string uniquely identifying this document version.
    """
    identity = f"{filename}|{hashlib.sha256(content).hexdigest()}"
    return hashlib.sha256(identity.encode()).hexdigest()[:16]


def ingest_pdf(
    file_bytes: bytes,
    filename: str,
    api_key: str,
) -> tuple[str, int]:
    """
    Ingest a PDF file into the Chroma vector store.

    Loads the PDF, splits it into overlapping chunks, attaches
    document-level metadata, embeds each chunk, and persists
    the result to disk. Returns the document ID and chunk count.

    Args:
        file_bytes: Raw bytes of the uploaded PDF.
        filename: Original filename, stored in chunk metadata.
        api_key: Google API key for embeddings.

    Returns:
        A tuple of (doc_id, chunks_created).
    """
    doc_id = compute_doc_id(filename, file_bytes)

    # Write bytes to a temp file for PyPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Attach metadata
    for page in pages:
        page.metadata["doc_id"] = doc_id
        page.metadata["source"] = filename

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(pages)

    # Assign chunk indices
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    # Embed and persist
    embeddings = build_embeddings(api_key)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    return doc_id, len(chunks)


def query_documents(
    question: str,
    api_key: str,
    llm,
    k: int = 4,
) -> tuple[str, List[SourceReference]]:
    """
    Retrieve relevant chunks and generate a grounded answer.

    Loads the persisted vector store, retrieves the top-k chunks
    most semantically similar to the question, builds a context
    string, and prompts the LLM to answer using only that context.

    Args:
        question: The user's natural-language question.
        api_key: Google API key for embeddings.
        llm: The language model instance.
        k: Number of chunks to retrieve.

    Returns:
        A tuple of (answer_string, list_of_SourceReference).

    Raises:
        FileNotFoundError: If no documents have been ingested yet.
    """
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            "No documents indexed yet. Please ingest a PDF first."
        )

    embeddings = build_embeddings(api_key)
    vector_store = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    # Retrieve relevant chunks
    results = vector_store.similarity_search(question, k=k)

    if not results:
        return "No relevant content found in the indexed documents.", []

    # Build context from retrieved chunks
    context = "\n\n---\n\n".join(
        f"[Page {doc.metadata.get('page', 0) + 1}]\n{doc.page_content}"
        for doc in results
    )

    # Prompt the LLM
    prompt = (
        f"You are a helpful assistant. Answer the question using ONLY "
        f"the context below. If the answer is not in the context, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    from langchain_core.messages import HumanMessage
    response = llm.invoke([HumanMessage(content=prompt)])
    answer = response.content

    # Build source references
    sources = []
    for doc in results:
        sources.append(SourceReference(
            page=doc.metadata.get("page", 0) + 1,
            filename=doc.metadata.get("source", "unknown"),
            chunk_index=doc.metadata.get("chunk_index", 0),
        ))

    return answer, sources


def get_store_stats(api_key: str) -> tuple[int, int]:
    """
    Return the number of unique documents and total chunks in the store.

    Args:
        api_key: Google API key for embeddings.

    Returns:
        A tuple of (unique_documents, total_chunks). Returns (0, 0)
        if the vector store has not been initialised yet.
    """
    if not CHROMA_DIR.exists():
        return 0, 0

    embeddings = build_embeddings(api_key)
    vector_store = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    collection = vector_store.get()
    total_chunks = len(collection["ids"])

    unique_docs = len(set(
        m.get("doc_id", "") for m in collection["metadatas"]
        if m
    ))

    return unique_docs, total_chunks