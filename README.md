# Fastdoc

Fastdoc is a multi–autonomous-agent information retrieval and RAG (Retrieval-Augmented Generation) system built with **LangGraph**, **Streamlit**, and **LLM-based agents**. The project provides multiple backend agent orchestration strategies and several Streamlit frontends for experimenting with different interaction patterns.

This repository focuses on document-centric question answering and information retrieval using coordinated agents.

---

## Features

* Multi-agent orchestration using **LangGraph**
* Retrieval-Augmented Generation (RAG)
* Multiple backend implementations (tool-based, MCP, database-backed, streaming, etc.)
* Streamlit-based interactive UI
* Local SQLite-based persistence
* Metrics collection for evaluation

---

## Project Structure

```
Fastdoc--A-multi-autonomous-agent-for-information-retrieval-main/
│
├── langgraph_backend.py              # Core LangGraph agent backend
├── langgraph_database_backend.py     # Database-backed LangGraph backend
├── langgraph_mcp_backend.py          # MCP-based agent backend
├── langgraph_tool_backend.py         # Tool-augmented agent backend
├── langraph_rag_backend.py           # RAG-specific backend implementation
│
├── streamlit_rag_frontend.py         # ✅ Main RAG Streamlit application
├── streamlit_frontend.py             # Basic Streamlit frontend
├── streamlit_frontend_database.py    # Database-based UI
├── streamlit_frontend_mcp.py          # MCP frontend
├── streamlit_frontend_streaming.py   # Streaming responses frontend
├── streamlit_frontend_threading.py   # Threaded execution frontend
├── streamlit_frontend_tool.py        # Tool-based frontend
│
├── metrics.py                        # Metrics and evaluation utilities
├── requirements.txt                  # Python dependencies
│
├── chatbot.db                        # SQLite database
├── chatbot.db-shm
├── chatbot.db-wal
└── README.md
```

---

## Requirements

* Python **3.10+**
* pip

All required Python packages are listed in `requirements.txt`.

---

## Installation

1. **Clone the repository**

```bash
git clone <repository-url>
cd Fastdoc--A-multi-autonomous-agent-for-information-retrieval-main
```

2. **Create and activate a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Running the Application

The primary application for running the Retrieval-Augmented Generation (RAG) system is:

```bash
streamlit run streamlit_rag_frontend.py
```

Once started, Streamlit will provide a local URL (typically `http://localhost:8501`) where you can interact with the application through your browser.

---

## Available Frontends

While `streamlit_rag_frontend.py` is the main entry point, the repository also includes alternative frontends for experimentation:

* `streamlit_frontend.py` – Basic UI
* `streamlit_frontend_database.py` – Database-backed interactions
* `streamlit_frontend_mcp.py` – MCP-based agent UI
* `streamlit_frontend_streaming.py` – Streaming responses
* `streamlit_frontend_threading.py` – Multi-threaded execution
* `streamlit_frontend_tool.py` – Tool-augmented agents

Each frontend can be run using:

```bash
streamlit run <frontend_file>.py
```

---

## Backend Variants

Fastdoc supports multiple backend agent orchestration strategies:

* **LangGraph Core Backend**
* **RAG Backend** for document-based QA
* **Tool-based Backend** for external tool usage
* **MCP Backend** for protocol-driven orchestration
* **Database Backend** for persistent agent state

These backends are modular and can be swapped depending on the frontend used.

---

## Data & Persistence

* Uses **SQLite** (`chatbot.db`) for storing conversation and agent state
* WAL and SHM files are automatically managed by SQLite

---

## Metrics

The `metrics.py` module provides utilities for tracking and evaluating agent performance and system behavior.

---

## Notes

* Ensure any required environment variables (e.g., API keys for LLM providers) are set before running the application.
* The project is intended for research and experimentation with multi-agent RAG systems.

---

## License

This project is provided as-is for research and educational purposes. License details should be added if distribution is intended.
