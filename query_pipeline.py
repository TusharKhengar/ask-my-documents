"""The 'R' and the 'G' of RAG: Retrieve relevant chunks, then Generate an answer.

Run ingestion_pipeline.py first - this file only reads the vector store.
"""

import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Reusing the SAME embedding function the store was built with. This import is
# the whole reason get_embedding_model() exists as a function.
from ingestion_pipeline import get_embedding_model

load_dotenv()


# ---------------------------------------------------------------- STEP 1
def load_vector_store(persist_directory="vector_store"):
    """Open the database of chunks that ingestion_pipeline.py built."""

    if not os.path.exists(persist_directory):
        raise FileNotFoundError(
            f"No vector store at '{persist_directory}'. "
            "Run 'python ingestion_pipeline.py' first."
        )

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=get_embedding_model(),
    )

    count = vectorstore._collection.count()
    if count == 0:
        raise ValueError("Vector store is empty. Re-run ingestion_pipeline.py.")

    print(f"Loaded vector store: {count} chunks.")
    return vectorstore


# ---------------------------------------------------------------- STEP 2
def retrieve(vectorstore, question, k=4):
    """Find the k chunks whose meaning sits closest to the question.

    The question is embedded with the same model as the chunks, then Chroma
    compares vectors. Score is cosine DISTANCE: lower means more similar.
    """

    results = vectorstore.similarity_search_with_score(question, k=k)

    print(f"\nRetrieved {len(results)} chunks:")
    for i, (doc, score) in enumerate(results, 1):
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"  [{i}] distance={score:.4f} | {preview}...")

    return [doc for doc, _ in results]


# ---------------------------------------------------------------- STEP 3
def format_context(docs):
    """Flatten the retrieved chunks into one labelled string for the prompt."""

    return "\n\n".join(
        f"[Source {i}] {doc.page_content}" for i, doc in enumerate(docs, 1)
    )


# ---------------------------------------------------------------- STEP 4
# The instructions wrapped around the retrieved text. The "say you don't know"
# line is what stops the model inventing facts that aren't in your documents.
PROMPT = ChatPromptTemplate.from_template(
    """You are answering questions using only the context below.

If the context does not contain the answer, say "I don't know based on the
provided documents." Do not use outside knowledge. Cite the source numbers
you used, like [Source 2].

Context:
{context}

Question: {question}

Answer:"""
)


# ---------------------------------------------------------------- STEP 5
def get_llm():
    """The model that writes the final answer.

    Reads GOOGLE_API_KEY from .env automatically. temperature=0 keeps answers
    grounded and repeatable instead of creative.
    """

    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set.\n"
            "  1. Get a free key at https://aistudio.google.com/apikey\n"
            "  2. Add this line to your .env file:\n"
            "     GOOGLE_API_KEY=your_key_here"
        )

    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.7-flash"),
        temperature=0,
    )


# ---------------------------------------------------------------- STEP 6
def answer_question(vectorstore, llm, question, k=4):
    """Tie steps 2-5 together: retrieve -> format -> prompt -> generate."""

    docs = retrieve(vectorstore, question, k=k)
    context = format_context(docs)

    messages = PROMPT.format_messages(context=context, question=question)
    response = llm.invoke(messages)

    # Gemini 3.x returns .content as a list of blocks (reasoning, text, ...),
    # older models return a plain string. .text flattens both to the answer.
    answer = response.text

    return answer, docs


def main():
    vectorstore = load_vector_store()
    llm = get_llm()

    print("\nAsk a question about your documents. Type 'exit' to quit.")

    while True:
        question = input("\n> ").strip()

        if question.lower() in {"exit", "quit", ""}:
            break

        answer, docs = answer_question(vectorstore, llm, question)

        print(f"\nAnswer:\n{answer}")
        print(f"\nBased on {len(docs)} chunks from: "
              f"{sorted({d.metadata.get('source', '?') for d in docs})}")


if __name__ == "__main__":
    main()
