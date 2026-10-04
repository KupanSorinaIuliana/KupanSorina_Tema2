from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader
)

# Registry: extensie → loader potrivit
LOADER_REGISTRY = {
    "pdf":  PyPDFLoader,
    "docx": Docx2txtLoader,
    "txt":  TextLoader
}

def load_document(path: str):
    ext = Path(path).suffix.lstrip(".").lower()
    if ext not in LOADER_REGISTRY:
        raise ValueError(f"Tip nesuportat: .{ext}")
    return LOADER_REGISTRY[ext](path).load()