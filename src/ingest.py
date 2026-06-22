"""
ingest.py - RAG Document Q&A Bot: Document Ingestion Pipeline

Loads PDF files from the data/ folder, splits them into chunks,
generates embeddings, and stores them in a Chroma vector database.
"""

import os
import glob
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR = "./data"                                   # Folder containing PDF files
VECTORSTORE_DIR = "./vectorstore"                     # Chroma persistence directory
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ── Core function ──────────────────────────────────────────────────────────────

def build_vector_store() -> Chroma:
    """
    Full ingestion pipeline:
      1. Discover and load all PDFs from DATA_DIR.
      2. Split documents into overlapping chunks.
      3. Embed chunks with a HuggingFace sentence-transformer.
      4. Persist the resulting Chroma vector store to VECTORSTORE_DIR.

    Returns:
        A ready-to-query Chroma vector store instance.

    Raises:
        FileNotFoundError: If DATA_DIR does not exist or contains no PDFs.
    """

    # ── Step 1: Discover PDF files ─────────────────────────────────────────────
    pdf_pattern = os.path.join(DATA_DIR, "**", "*.pdf")
    pdf_paths = sorted(glob.glob(pdf_pattern, recursive=True))

    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in '{DATA_DIR}'. "
            "Create the folder and add at least one PDF before running ingest."
        )

    print(f"\n📂  Found {len(pdf_paths)} PDF file(s) in '{DATA_DIR}':")
    for path in pdf_paths:
        print(f"     • {path}")

    # ── Step 2: Load documents via PyPDFLoader ─────────────────────────────────
    all_docs = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        # Enrich metadata: store the source filename and page number
        for page in pages:
            page.metadata["source"] = str(Path(pdf_path).name)
            # PyPDFLoader already sets page.metadata["page"] (0-indexed);
            # expose it also as a human-friendly 1-indexed field.
            page.metadata["page_number"] = page.metadata.get("page", 0) + 1

        all_docs.extend(pages)

    print(f"\n📄  Loaded {len(all_docs)} page(s) across all documents.")

    # ── Step 3: Split into chunks ──────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,   # Stores char offset in metadata for traceability
    )
    chunks = splitter.split_documents(all_docs)

    print(f"✂️   Created {len(chunks)} chunk(s) "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    # ── Step 4: Build embeddings ───────────────────────────────────────────────
    print(f"\n🤖  Loading embedding model: '{EMBEDDING_MODEL}' …")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},   # Switch to "cuda" if a GPU is available
        encode_kwargs={"normalize_embeddings": True},
    )

    # ── Step 5: Create & persist Chroma vector store ───────────────────────────
    print(f"💾  Indexing chunks into Chroma at '{VECTORSTORE_DIR}' …")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR,
        collection_name="rag_documents",
    )

    print(f"\n✅  Indexing complete — {len(chunks)} chunk(s) stored in "
          f"'{VECTORSTORE_DIR}'.\n")

    return vector_store


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_vector_store()