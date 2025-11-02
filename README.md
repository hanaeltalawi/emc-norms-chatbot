# EMC Norms Chatbot

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
- [File Structure](#file-structure)
- [About](#about)

---

## Features

- **Document Upload**
  - Upload and process `.docx` technical or regulatory documents.
  
- **Hybrid Search**
  - Combines **semantic embeddings (Sentence Transformers)** and **BM25 keyword scoring** for better retrieval accuracy.

- **Multiple Local Models via Ollama**
  - Choose between:
    - `llama3.1:8b`
    - `phi3:latest`
    - `phi:latest`
    - `mistral:7b`
    - `tinyllama:latest`

- **LangChain Integration**
  - Handles prompt chains, retrieval pipelines, and model orchestration.

- **Chroma Vector Database**
  - Efficient storage and querying of document embeddings.

- **Streamlit Frontend**
  - Clean, responsive chat interface for interactive Q&A sessions.

- **Runs Entirely Locally**
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
   git clone https://github.com/hanaeltalawi/emc-norms-chatbot.git
   cd emc-norms-chatbot
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama** 
  **macOS/Linux:**
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

    **Windows:**
    Download and install manually from the official site:
    https://ollama.ai

5. **Start the Ollama service** 
  Open a new terminal and run:
   ```bash
   ollama serve
   ```

6. **Pull the required models** 
  In a separate terminal window, download the models your app can use:
   ```bash
   ollama pull llama3.1:8b
   ollama pull phi3:latest  
   ollama pull phi:latest
   ollama pull mistral:7b
   ollama pull tinyllama:latest
   ```
    You can skip any model you don’t plan to use — just make sure at least one model is available locally.

7. **Launch the Streamlit app** 
  Finally, start the chatbot interface:
   ```bash
   streamlit run app.py
   ```
    This will automatically open your browser at http://localhost:8501 where you can upload a document and begin chatting.


## Configuration
The **EMC Norms Chatbot** runs locally and requires minimal configuration.  
By default, it uses in-memory settings and local paths, so no external API keys are needed.

## Usage
Once installed and the models are pulled, running the chatbot is straightforward:

1. **Start Ollama Service** (if not already running):
   ```bash
   ollama serve
   ```

2. **Launch the Streamlit web app:**
   ```bash
   streamlit run app.py
   ```
   
3. **Open your browser:**
   Streamlit will automatically open a new tab, usually at http://localhost:8501

4. **In the web interface:**
- Click "Upload Document" and select a .docx file.
- Choose one of the available local models:
  - llama3.1:8b
  - phi3:latest
  - phi:latest
  - mistral:7b
  - tinyllama:latest
- Type or speak your question about the document’s content.
- The chatbot will retrieve relevant context and generate a detailed response.

## File Structure
```
   emc-norms-chatbot/
   ├── app.py                         # Main Streamlit application
   ├── document_processor.py          # Handles DOCX parsing and text extraction
   ├── hybrid_search.py               # Implements BM25 + embedding-based retrieval
   ├── model_manager.py               # Manages Ollama model connections and responses
   ├── vector_store_manager.py        # Handles vector database creation and search (Chroma)
   ├── query_filter.py                # Utility for filtering and cleaning queries
   ├── standalone_model_comparison.py # Optional script for testing multiple models
   ├── requirements.txt               # Python dependencies
   └── README.md                      # Project documentation
```

## About
EMC Norms Chatbot is designed as a local, privacy-preserving AI assistant for understanding and analyzing Electromagnetic Compatibility (EMC) documents and standards.
Unlike cloud-based assistants, this system runs entirely offline, offering full control, reproducibility, and data privacy.
