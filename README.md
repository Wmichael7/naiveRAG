# Naive RAG

A simple Retrieval-Augmented Generation (RAG) implementation using LangChain, OpenAI embeddings, and FAISS for vector storage.

## Features

- **Document Ingestion**: Supports PDF, TXT, DOCX, Markdown, and web URLs
- **Semantic Chunking**: Uses OpenAI embeddings for intelligent document splitting
- **Vector Storage**: FAISS index for fast similarity search
- **RAG Pipeline**: Question-answering with source attribution
- **Evaluation**: RAGAS metrics for RAG quality assessment


## Setup

### Prerequisites

- Python 3.8+
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/naive-rag.git
   cd naive-rag
   ```

2. **Create virtual environment and install dependencies:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   uv sync
   ```

3. **Set up environment variables:**
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

### 1. Ingest Documents

Place your documents in the `data/` directory, then run:

```bash
uv run src/ingest.py
```

This will:
- Load documents from `data/`
- Create semantic chunks using OpenAI embeddings
- Save chunks to `artifacts/chunks.pkl`

### 2. Build Vector Index

```bash
uv run src/build_index.py
```

This creates a FAISS index from the chunks and saves it to `faiss_index/`.

### 3. Ask Questions

```bash
uv run src/rag_qa.py
```

Or modify the question in `src/rag_qa.py` and run it.

### 4. Evaluate RAG Quality (Optional)

```bash
uv run src/evaluate.py
```

Uses RAGAS metrics to evaluate answer faithfulness and relevance.

## Supported File Types

- **PDF**: `.pdf` files
- **Text**: `.txt` files
- **Word**: `.docx` files
- **Markdown**: `.md`, `.markdown` files
- **Web**: `.urls` files (one URL per line)

