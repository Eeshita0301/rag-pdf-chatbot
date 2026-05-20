# 🤖 RAG PDF + Image Chatbot

An AI-powered chatbot that lets you upload PDFs and images and ask questions about them using Retrieval Augmented Generation (RAG).

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-latest-green)
![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red?logo=streamlit)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1-orange)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector_db-purple)

---

## 🌐 Live Demo

👉 **[Try the app here](https://rag-pdf-chatbot-mcw9phr7frsyjac5z7hpds.streamlit.app/)

---

## 📌 What is RAG?

RAG (Retrieval Augmented Generation) is a technique where:
1. A document is split into chunks and converted to embeddings
2. When a question is asked, the most relevant chunks are retrieved
3. The retrieved chunks are passed to an LLM to generate an accurate answer

This prevents hallucination and ensures answers are grounded in your document.

---

## ✨ Features

- 📄 **PDF Upload** — Upload any PDF and chat with its content
- 🖼️ **Image Analysis** — Upload images and ask questions about them
- 🔍 **Semantic Search** — Finds relevant content by meaning, not just keywords
- 🧠 **Image Extraction from PDFs** — Automatically extracts and describes images inside PDFs
- 📌 **Source Citations** — Shows which page the answer came from
- 💬 **Chat History** — Maintains conversation context
- ⚡ **Fast Responses** — Powered by Groq's ultra-fast inference

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| LangChain | RAG pipeline orchestration |
| ChromaDB | Vector database for embeddings |
| HuggingFace (`all-MiniLM-L6-v2`) | Free local embedding model |
| Groq (LLaMA 3.1) | LLM for answer generation |
| Groq (LLaMA 4 Scout) | Vision model for image analysis |
| PyMuPDF (fitz) | Image extraction from PDFs |
| Streamlit | Web UI |
| Python-dotenv | Environment variable management |

---

## 🏗️ Architecture

```
PDF Upload
    ↓
Parse text (PyPDFLoader)
    ↓
Extract images (PyMuPDF) → Vision AI describes images
    ↓
Split into chunks (RecursiveCharacterTextSplitter)
    ↓
Convert to embeddings (HuggingFace all-MiniLM-L6-v2)
    ↓
Store in ChromaDB (Vector Database)
    ↓
User asks a question
    ↓
Question → embedding → semantic search → top 6 chunks
    ↓
Chunks + question → Groq LLaMA 3.1 → Answer + Sources
```

---

## 🚀 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Eeshita0301/rag-pdf-chatbot.git
cd rag-pdf-chatbot
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the root folder:
```
GROQ_API_KEY=your-groq-api-key-here
```
Get a free Groq API key at [console.groq.com](https://console.groq.com)

### 5. Run the app
```bash
streamlit run app/main.py
```

---

## 📁 Project Structure

```
rag-pdf-chatbot/
├── app/
│   ├── main.py        ← Streamlit web UI
│   ├── ingest.py      ← PDF ingestion pipeline
│   └── chat.py        ← Query pipeline (terminal version)
├── data/              ← Place your PDFs here
├── db/                ← ChromaDB vector store (auto-created)
├── .env               ← API keys (never commit this)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 💡 How to Use

1. Open the live app or run locally
2. Upload a PDF using the sidebar
3. Click **"Process PDF"** and wait for confirmation
4. Optionally upload an image
5. Type your question in the chat box
6. Get AI-powered answers with source citations!

---

## 🧠 Key Learnings

- Built a complete RAG pipeline from scratch
- Implemented semantic search using vector embeddings
- Tuned chunk size and overlap for optimal retrieval
- Identified and fixed LLM hallucination using strict prompts
- Integrated vision AI for multimodal document understanding
- Deployed a production AI application for free

---

## 🔮 Future Improvements

- [ ] Add conversation memory across sessions
- [ ] Support for multiple PDF uploads
- [ ] Add a document summarization feature
- [ ] Support for Excel and Word documents
- [ ] Add user authentication

---

## 👩‍💻 Author

**Eeshita** — [GitHub](https://github.com/Eeshita0301)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
