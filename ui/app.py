"""
AI Document Intelligence — Streamlit Frontend

A browser-based UI that communicates with the FastAPI backend.
Run alongside the API using docker-compose, or start manually.

Usage:
    streamlit run ui/app.py
"""
import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="📄",
    layout="wide",
)

st.markdown("""
<style>
.title  {font-size:2rem;font-weight:700;color:#1A5276;}
.sub    {font-size:1rem;color:#717D7E;margin-bottom:1rem;}
.answer {background:#EBF5FB;padding:14px;border-radius:8px;border-left:4px solid #2874A6;}
.source {background:#F2F3F4;padding:8px 12px;border-radius:6px;margin:4px 0;font-size:0.85rem;}
.badge-ok   {background:#E9F7EF;color:#1E8449;padding:4px 10px;border-radius:12px;font-size:0.8rem;}
.badge-err  {background:#FDEDEC;color:#C0392B;padding:4px 10px;border-radius:12px;font-size:0.8rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">📄 AI Document Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p class="sub">Upload PDFs · Index them · Ask questions · Get grounded answers with source citations</p>', unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_status() -> dict | None:
    try:
        r = requests.get(f"{API_BASE}/status", timeout=5)
        return r.json() if r.status_code == 200 else None
    except requests.RequestException:
        return None


def ingest_pdf(file_bytes: bytes, filename: str) -> dict | None:
    try:
        r = requests.post(
            f"{API_BASE}/ingest",
            files={"file": (filename, file_bytes, "application/pdf")},
            timeout=120,
        )
        return r.json()
    except requests.RequestException as e:
        return {"detail": str(e)}


def query_api(question: str, k: int) -> dict | None:
    try:
        r = requests.post(
            f"{API_BASE}/query",
            json={"question": question, "k": k},
            timeout=60,
        )
        return r.json()
    except requests.RequestException as e:
        return {"detail": str(e)}


# ── Sidebar: status + upload ──────────────────────────────────────────────────
with st.sidebar:
    st.subheader("⚙️ API Status")

    status_data = get_status()
    if status_data:
        st.markdown(
            f'<span class="badge-ok">✅ Connected</span>',
            unsafe_allow_html=True,
        )
        st.metric("Documents indexed", status_data.get("documents_indexed", 0))
        st.metric("Chunks indexed", status_data.get("chunks_indexed", 0))
        st.caption(f"Model: {status_data.get('model', 'unknown')}")
    else:
        st.markdown(
            '<span class="badge-err">❌ API not reachable</span>',
            unsafe_allow_html=True,
        )
        st.caption("Make sure the FastAPI server is running on port 8000.")

    st.divider()
    st.subheader("📤 Upload PDF")

    uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded:
        if st.button("🔄 Ingest Document", type="primary"):
            with st.spinner(f"Ingesting {uploaded.name}..."):
                result = ingest_pdf(uploaded.read(), uploaded.name)

            if result and "doc_id" in result:
                st.success(
                    f"✅ Ingested!\n\n"
                    f"**File:** {result['filename']}\n\n"
                    f"**Chunks:** {result['chunks_created']}\n\n"
                    f"**ID:** `{result['doc_id']}`"
                )
                st.rerun()
            else:
                st.error(f"❌ {result.get('detail', 'Unknown error')}")

    st.divider()

    k_val = st.slider(
        "Chunks to retrieve (k)",
        min_value=1, max_value=10, value=4,
        help="Higher k = more context, slower response",
    )

    if st.button("🗑️ Reset vector store", type="secondary"):
        try:
            r = requests.delete(f"{API_BASE}/reset", timeout=10)
            st.success(r.json().get("message", "Done"))
            st.rerun()
        except Exception as e:
            st.error(str(e))


# ── Main panel: Q&A ──────────────────────────────────────────────────────────
st.subheader("💬 Ask a Question")

if not status_data:
    st.warning("Cannot connect to the API. Start the FastAPI server first.")
elif status_data.get("documents_indexed", 0) == 0:
    st.info("👈 Upload and ingest at least one PDF to get started.")
else:
    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for entry in st.session_state.chat_history:
        st.markdown(f"**🧑 You:** {entry['question']}")
        st.markdown(
            f'<div class="answer">🤖 {entry["answer"]}</div>',
            unsafe_allow_html=True,
        )
        if entry["sources"]:
            with st.expander("📎 Sources"):
                for src in entry["sources"]:
                    st.markdown(
                        f'<div class="source">📄 <b>{src["filename"]}</b> — '
                        f'Page {src["page"]} (chunk {src["chunk_index"]})</div>',
                        unsafe_allow_html=True,
                    )
        st.divider()

    question = st.chat_input("Ask anything about your documents...")
    if question:
        with st.spinner("Thinking..."):
            result = query_api(question, k_val)

        if result and "answer" in result:
            st.session_state.chat_history.append({
                "question": question,
                "answer": result["answer"],
                "sources": result.get("sources", []),
            })
            st.rerun()
        else:
            st.error(f"❌ {result.get('detail', 'Query failed')}")