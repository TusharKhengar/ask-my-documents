import os
import shutil
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The specified path '{docs_path}' does not exist.")

    loader = DirectoryLoader(
        path=docs_path, glob="*.txt", loader_cls=TextLoader) 
    
    documents = loader.load()

    if len(documents) == 0:
        raise ValueError(f"No documents found in the specified path '{docs_path}'.")

    for i, doc in enumerate(documents):
        print(f"Document {i+1}: {doc.metadata['source']} - {len(doc.page_content)} characters {doc.page_content[:10]}... {doc.metadata}")

    return documents



def split_documents(documents, chunk_size=800, chunk_overlap=20):

    # Recursive splitter tries separators in order: paragraph -> line ->
    # sentence -> word. If a paragraph is too big it falls through to the next
    # separator instead of emitting an oversized chunk, so chunk_size is a real
    # limit here rather than a suggestion.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Split {len(documents)} document(s) into {len(chunks)} chunks.")
    print(f"Largest chunk: {max(len(c.page_content) for c in chunks)} characters.")

    # if chunks:
    #     for i, chunk in enumerate(chunks[:5]):  # Print first 5 chunks as an example
    #         print(f"Split Document {i}")
    #         print(f"Source: {chunk.metadata['source']}")
    #         print(f"Characters: {len(chunk.page_content)}")
    #         print(f"Content: {chunk.page_content[:100]}...")
    #         print(f"Metadata: {chunk.metadata}")
    #         print("-" * 40)

    #     if len(chunks) > 5:
    #         print(f"...and {len(chunks) - 5} more chunks.")


    return chunks

def get_embedding_model():
    """The one place that decides how text becomes vectors.

    BAAI/bge-small-en-v1.5 runs locally through ONNX: no API key, no usage
    limits, works offline after the ~130 MB model is downloaded on first run.
    It outputs 384 numbers per chunk (Voyage produced 1024, which is why any
    old vector store has to be rebuilt from scratch).

    Anything that searches this vector store later must call THIS function too
    — a store built with one model can only be queried with the same model.
    """
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def create_vector_store(chunks, persist_directory="vector_store"):

    embedding_model = get_embedding_model()

    # Chroma appends to whatever is already on disk, and it will crash if those
    # old vectors have a different length. Start clean.
    if os.path.exists(persist_directory):
        print(f"Removing existing vector store at '{persist_directory}'...")
        shutil.rmtree(persist_directory)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print(f"Vector store created and persisted at '{persist_directory}'.")
    return vectorstore



def main():

    documents = load_documents(docs_path="docs")

    chunks = split_documents(documents)  

    vectorstore = create_vector_store(chunks)

if __name__ == "__main__":
    main()
    
   