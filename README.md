# 📚 RAG Document Q&A Bot

A Retrieval-Augmented Generation (RAG) based AI system that allows users to ask questions over multiple documents and get accurate, context-aware answers with citations.

---

## 🚀 Features
- Loads multiple PDF/TXT documents
- Splits documents into chunks with overlap
- Generates embeddings using LLM embedding model
- Stores embeddings in vector database (ChromaDB/FAISS)
- Retrieves top-k relevant chunks using similarity search
- Generates grounded answers using LLM (OpenAI / Gemini)
- Displays source citations (file + chunk context)
- CLI-based interactive Q&A system

---

## 🧠 Architecture
Document → Chunking → Embedding → Vector DB → Retrieval → LLM Answer

---

## 📂 Project Structure

rag-document-bot/
│
├── data/ # PDF documents
├── src/
│ ├── ingest.py # document loading & indexing
│ ├── query.py # retrieval logic
│ ├── rag.py # core pipeline
│ ├── utils.py # helpers
│
├── vectorstore/ # stored embeddings
├── main.py # entry point
├── requirements.txt
├── .gitignore
└── README.md


---

## ⚙️ Installation

```bash
git clone https://github.com/BHAGYASREE-RANGU/rag-document-bot.git
cd rag-document-bot

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt

---

# 🚀 STEP 2: Add missing files check

Make sure you have:

✔ requirements.txt  
✔ .gitignore  
✔ data/ (PDFs)  
✔ src/ (all python files)  
✔ main.py  

---

# 🚀 STEP 3: requirements.txt (important)

If not sure, use this:

```txt
openai
chromadb
pypdf
tiktoken
langchain
python-dotenv
