# imports
from pathlib import Path
from dotenv import load_dotenv
from typing import Callable, Dict, Any

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# loaders to access various file types
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
    WebBaseLoader,
)

# new embeddings and chunking
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

# table of file extensions to loader classes
LOADER_MAP: Dict[str, Callable[[str], Any]] = {
    ".txt": TextLoader,
    ".pdf": PyPDFLoader,
    ".md": UnstructuredMarkdownLoader,
    ".markdown": UnstructuredMarkdownLoader,
    ".docx": Docx2txtLoader,
    ".urls": WebBaseLoader,
}


def dispatch_loader(path: Path):
    ext = path.suffix.lower()
    loader_cls = LOADER_MAP.get(ext, TextLoader)
    if loader_cls is None:
        loader_cls = TextLoader
    if loader_cls == TextLoader:
        return TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
    return loader_cls(str(path))


# ingest files from the data directory
RAW_DIR = Path("data")
docs = []

for fp in RAW_DIR.rglob("*"):
    if fp.is_file() and not fp.name.startswith("."):
        docs += dispatch_loader(fp).load()
        print("✅ loaded", fp.name)

# semantic chunking
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
splitter = SemanticChunker(embeddings, buffer_size=3)
chunks = splitter.split_documents(docs)
all_paths = list(RAW_DIR.rglob("*"))
print(f"{len(all_paths)} files → {len(chunks)} semantic chunks")


# indexing
import pickle, os

os.makedirs("artifacts", exist_ok=True)
pickle.dump(chunks, open("artifacts/chunks.pkl", "wb"))
