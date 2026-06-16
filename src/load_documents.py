"""Load PDF and TXT documents from local paths."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


def load_documents(file_paths: list[str | Path]) -> list[Document]:
    """Load documents from PDF or TXT files."""
    documents: list[Document] = []

    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        elif suffix == ".txt":
            loader = TextLoader(str(path), encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {suffix}. Use .pdf or .txt")

        documents.extend(loader.load())

    return documents
