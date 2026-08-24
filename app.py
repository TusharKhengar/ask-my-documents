"""Streamlit front-end for the RAG pipeline.

Local:  streamlit run app.py
Hosted: Streamlit Community Cloud, with GOOGLE_API_KEY set in app secrets.
"""

# Streamlit Cloud's base image ships an sqlite3 too old for Chroma. Swap in the
# modern bundled build BEFORE anything imports chromadb. No-op on Windows.
try:
    __import__("pysqlite3")
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import streamlit as st

st.set_page_config(page_title="Ask My Documents", page_icon="📚", layout="centered")

# Streamlit Cloud injects secrets via st.secrets, not .env. Copy them into the
# environment so query_pipeline.py works unchanged in both places.
try:
    for _key in ("GOOGLE_API_KEY", "GEMINI_MODEL"):
        if _key in st.secrets and not os.getenv(_key):
            os.environ[_key] = st.secrets[_key]
except Exception:
    pass  # No secrets.toml locally - .env handles it.

from query_pipeline import load_vector_store, get_llm, answer_question


# Loading the embedding model and opening Chroma takes a few seconds. Without
# caching, Streamlit would redo it on every single keystroke and interaction.
@st.cache_resource(show_spinner="Loading knowledge base…")
def get_store():
    return load_vector_store()


@st.cache_resource(show_spinner="Connecting to Gemini…")
def get_model():
    return get_llm()


st.title("📚 Ask My Documents")
st.caption(
    "Retrieval-Augmented Generation — answers are grounded in the indexed "
    "documents, with citations. If the answer isn't in them, it says so."
)

with st.sidebar:
    st.header("Settings")
    k = st.slider(
        "Chunks to retrieve", min_value=2, max_value=10, value=4,
        help="How many document passages get fed to the model as context.",
    )
    show_chunks = st.checkbox("Show retrieved chunks", value=True)

    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "1. Your question is converted to a 384-dimension vector\n"
        "2. Chroma finds the closest chunks by cosine distance\n"
        "3. Those chunks are pasted into the prompt\n"
        "4. Gemini answers using only that context\n\n"
        "Embeddings run locally (BAAI/bge-small-en-v1.5) — no embedding API cost."
    )

# Fail loudly and usefully if the key or the store is missing.
try:
    store = get_store()
    llm = get_model()
except Exception as exc:
    st.error(f"**Startup failed**\n\n```\n{exc}\n```")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask something about the documents…")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents…"):
            try:
                answer, docs = answer_question(store, llm, question, k=k)
            except Exception as exc:
                answer, docs = f"Something went wrong:\n\n```\n{exc}\n```", []

        st.markdown(answer)

        if docs and show_chunks:
            with st.expander(f"Retrieved {len(docs)} chunks"):
                for i, doc in enumerate(docs, 1):
                    source = os.path.basename(doc.metadata.get("source", "unknown"))
                    st.markdown(f"**[Source {i}]** · `{source}`")
                    st.text(doc.page_content[:600])
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})
