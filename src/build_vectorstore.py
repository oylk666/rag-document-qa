"""Split documents, embed text, and store vectors in Chroma."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def get_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """Create a local HuggingFace embedding model."""
    cache_dir = Path(__file__).resolve().parent.parent / ".model_cache"
    common_kwargs = {
        "model_name": model_name,
        "cache_folder": str(cache_dir),
    }

    try:
        return HuggingFaceEmbeddings(
            **common_kwargs,
            model_kwargs={"local_files_only": True},
        )
    except Exception:
        return HuggingFaceEmbeddings(**common_kwargs)


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split long documents into smaller text chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vectorstore(
    documents: list[Document],
    persist_directory: str | Path,
    collection_name: str = "documents",
) -> Chroma:
    """Build a Chroma vector store from document chunks."""
    chunks = split_documents(documents)
    embeddings = get_embeddings()

    persist_path = Path(persist_directory)
    persist_path.mkdir(parents=True, exist_ok=True)

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_path),
    )


def load_vectorstore(
    persist_directory: str | Path,
    collection_name: str = "documents",
) -> Chroma:
    """Load an existing Chroma vector store."""
    embeddings = get_embeddings()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )
