"""
query.py - RAG Document Q&A Bot: Retrieval Pipeline

Loads an existing Chroma vector store and retrieves the most relevant
document chunks for a given natural-language query.
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# ── Configuration ──────────────────────────────────────────────────────────────

VECTORSTORE_DIR = "./vectorstore"                      # Must match ingest.py
COLLECTION_NAME = "rag_documents"                      # Must match ingest.py
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ── Initialise shared resources once at import time ────────────────────────────
# Both the embeddings model and the vector store are expensive to construct,
# so they are created as module-level singletons rather than inside the function.

print(f"🤖  Loading embedding model: '{EMBEDDING_MODEL}' …")
_embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},          # Switch to "cuda" for GPU inference
    encode_kwargs={"normalize_embeddings": True},
)

print(f"📦  Connecting to Chroma vector store at '{VECTORSTORE_DIR}' …")
_vector_store = Chroma(
    persist_directory=VECTORSTORE_DIR,
    embedding_function=_embeddings,
    collection_name=COLLECTION_NAME,
)

total_docs = _vector_store._collection.count()
print(f"✅  Vector store ready — {total_docs} chunk(s) indexed.\n")


# ── Core retrieval function ────────────────────────────────────────────────────

def retrieve_documents(query: str, k: int = 3) -> list[Document]:
    """
    Retrieve the top-k most semantically similar document chunks for *query*.

    Args:
        query: Natural-language question or search string.
        k:     Number of chunks to return (default: 3).

    Returns:
        A list of LangChain Document objects, each carrying:
          - page_content  : the raw text of the chunk
          - metadata      : dict with at minimum 'source' and 'page_number'

    Raises:
        ValueError: If the vector store contains no documents.
    """
    if total_docs == 0:
        raise ValueError(
            "The vector store is empty. Run ingest.py first to index your PDFs."
        )

    # Clamp k so we never ask for more results than exist in the store
    effective_k = min(k, total_docs)
    if effective_k < k:
        print(f"⚠️   Requested k={k} but only {total_docs} chunk(s) available; "
              f"returning {effective_k}.\n")

    results: list[Document] = _vector_store.similarity_search(
        query=query,
        k=effective_k,
    )

    return results


# ── Pretty-print helper ────────────────────────────────────────────────────────

def print_results(query: str, results: list[Document]) -> None:
    """
    Display retrieved chunks in a readable format.

    Args:
        query:   The original query string (printed as a header).
        results: List of Document objects returned by retrieve_documents().
    """
    print("=" * 70)
    print(f"🔍  Query : {query}")
    print(f"📊  Results: {len(results)} chunk(s) retrieved")
    print("=" * 70)

    if not results:
        print("  No matching chunks found.")
        return

    for rank, doc in enumerate(results, start=1):
        source      = doc.metadata.get("source", "unknown")
        page_number = doc.metadata.get("page_number", "n/a")
        start_index = doc.metadata.get("start_index", "n/a")
        content     = doc.page_content.strip()

        print(f"\n  ── Result #{rank} {'─' * 50}")
        print(f"  📄  Source      : {source}")
        print(f"  📖  Page number : {page_number}")
        print(f"  🔢  Char offset : {start_index}")
        print(f"\n  Content:\n")
        # Indent each line for visual separation
        for line in content.splitlines():
            print(f"    {line}")

    print("\n" + "=" * 70 + "\n")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Example test query — replace with your own ────────────────────────────
    TEST_QUERY = "What are the main topics covered in the documents?"

    retrieved = retrieve_documents(query=TEST_QUERY, k=3)
    print_results(query=TEST_QUERY, results=retrieved)