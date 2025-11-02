# ⚡ EMC Norms Chatbot

**EMC Norms Chatbot** is a local AI-powered document analysis and Q&A system built with **LangChain**, **Ollama**, and **Chroma**.  
It allows users to upload technical documents (such as EMC compliance standards), process them for context understanding, and interact in real-time via a **Streamlit** interface.  

The chatbot performs **hybrid retrieval** — combining **semantic embeddings** and **keyword-based (BM25)** searches — to deliver accurate, context-aware answers grounded in the uploaded content.

---

## Table of Contents

- [Features](#features)
- [Technologies](#technologies)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [System Architecture](#system-architecture)
- [File Structure](#file-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [About](#about)

---

## Features

- 📂 **Document Upload**
  - Upload and process `.docx` technical or regulatory documents.
  
- 🔍 **Hybrid Search**
  - Combines **semantic embeddings (Sentence Transformers)** and **BM25 keyword scoring** for better retrieval accuracy.

- 🤖 **Multiple Local Models via Ollama**
  - Choose between:
    - `llama3.1:8b`
    - `phi3:latest`
    - `phi:latest`
    - `mistral:7b`
    - `tinyllama:latest`

- 🧠 **LangChain Integration**
  - Handles prompt chains, retrieval pipelines, and model orchestration.

- ⚙️ **Chroma Vector Database**
  - Efficient storage and querying of document embeddings.

- 💬 **Streamlit Frontend**
  - Clean, responsive chat interface for interactive Q&A sessions.

- 🔒 **Runs Entirely Locally**
  - All model inference and document processing happen on your machine — ensuring privacy and full control.

---

## Technologies

- **Programming Language:** Python 3.12.7  
- **Frameworks & Libraries:**
  - `streamlit` – Web app interface
  - `langchain`, `langchain-community`, `langchain-core`, `langchain-huggingface`, `langchain-chroma` – AI orchestration and retrieval
  - `chromadb` – Vector database backend
  - `sentence-transformers` – Text embedding model
  - `rank-bm25` – Keyword-based scoring
  - `torch` – Model acceleration (CPU/GPU)
  - `python-docx` – Document parsing
  - `pandas`, `numpy` – Data handling
  - `requests`, `urllib3`, `pypika`, `typing-extensions` – Utility modules
- **Model Backend:** Ollama (local LLM runner)
- **Supported Models:** LLaMA 3.1, Phi, Phi-3, Mistral, TinyLLaMA

---

## Installation

### Prerequisites
- Python **3.12+**
- Ollama installed locally
- (Optional) CUDA-enabled GPU for acceleration

### Steps

1. **Clone this repository:**
   ```bash
   git clone https://github.com/yourusername/emc-norms-chatbot.git
   cd emc-norms-chatbot