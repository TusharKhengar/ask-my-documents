# 📚 Ask My Documents — RAG over a local corpus

A Retrieval-Augmented Generation system that answers questions about a document
collection it was never trained on, cites the passages it used, and says
**"I don't know"** when the answer isn't in the documents.

**[▶ Live demo](#)** · built with LangChain, ChromaDB, FastEmbed and Gemini

---

## What it does

Large language models only know their training data. This project makes one
answer questions about an arbitrary corpus — here, 6.4 MB of Project Gutenberg
text indexed into **11,304 searchable chunks** — without any fine-tuning.

```
                    INGESTION (run once)
   docs/*.txt ──► split into chunks ──► embed locally ──► ChromaDB
                    800 chars each        384-dim vectors    (on disk)

                    QUERY (every question)
   question ──► embed ──► find 4 nearest chunks ──► prompt ──► Gemini ──► answer
                            (cosine distance)      + context           + citations
```

## Why the retrieval matters

Asked *"Who is Irene Adler and why does Sherlock Holmes remember her?"*, the
retriever surfaces the passage beginning *"To Sherlock Holmes she is always the
woman"* — out of 11,304 candidates. The word "remember" never appears in that
passage. Semantic search matches on **meaning**, not keywords.

Asked *"What is the capital of Brazil?"*, the system replies **"I don't know
based on the provided documents."** Gemini plainly knows the answer; the prompt
forbids using knowledge outside the retrieved context. That refusal is the
feature — a RAG system that answers off-corpus questions is one that will also
invent confident nonsense about your actual documents.

## Design decisions

| Decision | Reasoning |
|---|---|
| **Local embeddings** (`BAAI/bge-small-en-v1.5` via FastEmbed) | Zero API cost and no rate limits on the highest-volume operation. ONNX runtime instead of PyTorch keeps the deploy footprint ~150 MB rather than ~2.5 GB. |
| **Single `get_embedding_model()`** shared by both pipelines | A vector store can only be queried by the model that built it. Mismatched models fail *silently* with bad results, so the model is defined exactly once. |
| **`RecursiveCharacterTextSplitter`** | Falls back paragraph → line → sentence → word, so `chunk_size` is a hard limit. The simpler `CharacterTextSplitter` emitted chunks 25% over the limit whenever a paragraph wouldn't split. |
| **Prebuilt index committed to the repo** | Embedding 11,304 chunks takes ~37 min of CPU. The host opens a prebuilt index instead of timing out on cold start. |
| **`temperature=0`** | The model should transcribe retrieved facts, not improvise. |
| **Grounding instruction in the prompt** | The single most important line in the project — it's what converts a fluent guesser into a citable system. |

## Tech stack

| Layer | Choice |
|---|---|
| Orchestration | LangChain |
| Vector store | ChromaDB (cosine distance) |
| Embeddings | FastEmbed · `BAAI/bge-small-en-v1.5` · 384-dim · runs locally |
| LLM | Google Gemini (`gemini-3.7-flash`) |
| UI | Streamlit |

## Running it locally

```bash
git clone https://github.com/TusharKhengar/ask-my-documents.git
cd ask-my-documents

python -m venv ragvenv
ragvenv\Scripts\activate          # Windows
# source ragvenv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Add a free [Google AI Studio key](https://aistudio.google.com/apikey) to `.env`:

```
GOOGLE_API_KEY=your_key_here
```

Then launch:

```bash
streamlit run app.py
```

The prebuilt index ships with the repo. To rebuild it from your own documents,
drop `.txt` files into `docs/` and run:

```bash
python ingestion_pipeline.py     # ~37 min for the full corpus
```

## Project layout

```
ingestion_pipeline.py   load → split → embed → persist to ChromaDB
query_pipeline.py       retrieve → build context → prompt → generate
app.py                  Streamlit chat UI with source inspection
vector_store/           prebuilt Chroma index (11,304 chunks)
docs/                   source corpus
```

`query_pipeline.py` runs standalone as a CLI too:

```bash
python query_pipeline.py
```

## Possible extensions

- Reranking retrieved chunks with a cross-encoder before they hit the prompt
- Hybrid search — BM25 keyword matching blended with vector similarity
- Streaming token-by-token responses
- Conversation memory so follow-up questions resolve pronouns
