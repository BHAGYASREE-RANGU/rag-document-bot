"""
rag.py - RAG Document Q&A Bot: Generation Pipeline

Retrieves relevant document chunks from the Chroma vector store and
passes them as grounded context to Google Gemini to produce an answer
that is strictly constrained to the provided documents.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# Import the retrieval function built in query.py
from query import retrieve_documents


# ── Environment setup ──────────────────────────────────────────────────────────

load_dotenv()  # Reads .env from the current working directory

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "GOOGLE_API_KEY is not set. "
        "Add it to a .env file in the project root:\n"
        "  GOOGLE_API_KEY=your_key_here"
    )

# ── Model configuration ────────────────────────────────────────────────────────

GEMINI_MODEL = "gemini-2.5-flash"
FALLBACK_RESPONSE = (
    "I could not find relevant information in the provided documents."
)

# ── Initialise the Gemini LLM ──────────────────────────────────────────────────

print(f"🤖  Initialising Gemini model: '{GEMINI_MODEL}' …\n")
_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0,        # Low temperature for factual, grounded answers
    max_output_tokens=1024,
)


# ── Prompt templates ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a precise document assistant. Your job is to answer the user's question \
using ONLY the context passages provided below.

Rules you must follow without exception:
1. Base your answer exclusively on the supplied context — never on outside knowledge.
2. If the context does not contain enough information to answer the question, \
respond with exactly: "{fallback}"
3. Be concise, factual, and direct.
4. Do not speculate, infer, or add information that is not explicitly stated \
in the context.
5. If quoting from the context, keep quotes short and accurate.
""".format(fallback=FALLBACK_RESPONSE)

_USER_PROMPT_TEMPLATE = """\
### Context
{context}

### Question
{question}

### Answer
"""


# ── Core RAG function ──────────────────────────────────────────────────────────

def answer_question(question: str, k: int = 3) -> dict:
    """
    Full RAG pipeline: retrieve → build context → generate answer.

    Args:
        question: Natural-language question from the user.
        k:        Number of document chunks to retrieve (default: 3).

    Returns:
        A dict with the shape:
        {
            "answer":  str,                        # Gemini's grounded answer
            "sources": [                           # Deduplicated source list
                {"source": str, "page": int|str},
                ...
            ]
        }

    Raises:
        EnvironmentError: Propagated from module-level key validation.
        Exception:        Any upstream LangChain / Gemini API error.
    """

    print(f"🔍  Retrieving top-{k} chunk(s) for: \"{question}\"")

    # ── Step 1: Retrieve relevant chunks ──────────────────────────────────────
    docs = retrieve_documents(query=question, k=k)

    if not docs:
        print("⚠️   No chunks retrieved — returning fallback response.")
        return {"answer": FALLBACK_RESPONSE, "sources": []}

    # ── Step 2: Build the context block ───────────────────────────────────────
    context_parts: list[str] = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page_number", "n/a")
        text   = doc.page_content.strip()

        # Label each passage so the model can reference them if needed
        context_parts.append(
            f"[Passage {i} | File: {source} | Page: {page}]\n{text}"
        )

    context_block = "\n\n".join(context_parts)

    # ── Step 3: Compose the prompt ─────────────────────────────────────────────
    user_prompt = _USER_PROMPT_TEMPLATE.format(
        context=context_block,
        question=question,
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    # ── Step 4: Call Gemini ────────────────────────────────────────────────────
    print(f"💬  Sending prompt to Gemini ({GEMINI_MODEL}) …")
    response = _llm.invoke(messages)
    answer_text: str = response.content.strip()
    print("✅  Answer received.\n")

    # ── Step 5: Deduplicate and collect sources ────────────────────────────────
    seen: set[tuple] = set()
    sources: list[dict] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page_number", "n/a")
        key    = (source, page)
        if key not in seen:
            seen.add(key)
            sources.append({"source": source, "page": page})

    return {
        "answer":  answer_text,
        "sources": sources,
    }


# ── Pretty-print helper ────────────────────────────────────────────────────────

def print_rag_result(question: str, result: dict) -> None:
    """
    Display a RAG result in a readable format.

    Args:
        question: The original question.
        result:   Dict returned by answer_question().
    """
    print("=" * 70)
    print(f"❓  Question : {question}")
    print("=" * 70)
    print(f"\n💡  Answer:\n\n    {result['answer']}\n")

    if result["sources"]:
        print("📚  Sources:")
        for s in result["sources"]:
            print(f"     • {s['source']}  (page {s['page']})")
    else:
        print("📚  Sources: none")

    print("\n" + "=" * 70 + "\n")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Example test question — replace with your own ─────────────────────────
    TEST_QUESTION = "What are the main topics covered in the documents?"

    result = answer_question(question=TEST_QUESTION, k=3)
    print_rag_result(question=TEST_QUESTION, result=result)