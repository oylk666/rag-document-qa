"""Streamlit app for local document Q&A with RAG."""

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.build_vectorstore import build_vectorstore, load_vectorstore
from src.load_documents import load_documents
from src.qa_chain import ask_question, create_llm, create_qa_chain

load_dotenv()

# Avoid tracing issues with non-ASCII text in some environments.
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = ROOT / "chroma_db"


def save_uploaded_files(uploaded_files) -> list[Path]:
    """Save uploaded files to a temp directory."""
    temp_dir = Path(tempfile.mkdtemp())
    saved_paths: list[Path] = []

    for uploaded_file in uploaded_files:
        file_path = temp_dir / uploaded_file.name
        file_path.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(file_path)

    return saved_paths


def get_default_documents() -> list[Path]:
    """Return sample documents from the data folder."""
    paths = []
    for pattern in ("*.pdf", "*.txt"):
        paths.extend(DATA_DIR.glob(pattern))
    return sorted(paths)


def main() -> None:
    st.set_page_config(page_title="Local Document QA", page_icon="📄", layout="wide")
    st.title("Local Document QA System Based on RAG")
    st.caption("基于 RAG 的本地文档智能问答系统")

    with st.sidebar:
        st.header("Settings")
        llm_provider = st.selectbox("LLM Provider", ["ollama", "openai"])
        model_name = st.text_input(
            "Model Name",
            value=os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            if llm_provider == "openai"
            else os.getenv("OLLAMA_MODEL", "gemma3:12b"),
        )
        top_k = st.slider("Retrieved Chunks (k)", min_value=1, max_value=8, value=4)
        st.markdown("---")
        st.markdown("**RAG Pipeline**")
        st.markdown("1. Load PDF/TXT\n2. Split text\n3. Embed vectors\n4. Store in Chroma\n5. Retrieve + Generate")

    tab_upload, tab_default = st.tabs(["Upload Documents", "Use Sample Documents"])

    with tab_upload:
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )

    with tab_default:
        default_docs = get_default_documents()
        if default_docs:
            st.write("Sample files in `data/`:")
            for doc in default_docs:
                st.write(f"- `{doc.name}`")
        else:
            st.warning("No sample documents found in `data/`.")

    build_clicked = st.button("Build / Rebuild Knowledge Base", type="primary")

    if build_clicked:
        file_paths: list[Path] = []

        if uploaded_files:
            file_paths = save_uploaded_files(uploaded_files)
        elif default_docs:
            file_paths = default_docs
        else:
            st.error("Please upload documents or add files to the data folder.")
            st.stop()

        with st.spinner("Loading documents, splitting text, and building vector store..."):
            try:
                documents = load_documents(file_paths)
                vectorstore = build_vectorstore(documents, CHROMA_DIR)
                st.session_state["vectorstore_ready"] = True
                st.session_state["doc_count"] = len(documents)
                st.success(f"Knowledge base built from {len(documents)} document page(s)/file(s).")
            except Exception as exc:
                st.error(f"Failed to build knowledge base: {exc}")
                st.stop()

    st.markdown("---")
    st.subheader("Ask a Question")

    question = st.text_input(
        "Enter your question",
        placeholder="例如：RAG 的核心流程是什么？",
    )

    if st.button("Get Answer") and question:
        if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
            st.warning("Please build the knowledge base first.")
            st.stop()

        try:
            with st.spinner("Retrieving relevant content and generating answer..."):
                vectorstore = load_vectorstore(CHROMA_DIR)
                llm = create_llm(llm_provider, model_name or None)
                chain = create_qa_chain(vectorstore, llm, k=top_k)
                result = ask_question(chain, question)

            st.markdown("### Answer")
            st.write(result["result"])

            source_docs = result.get("source_documents", [])
            if source_docs:
                with st.expander("Retrieved Source Chunks"):
                    for idx, doc in enumerate(source_docs, start=1):
                        source = doc.metadata.get("source", "unknown")
                        page = doc.metadata.get("page", "N/A")
                        st.markdown(f"**Chunk {idx}** — `{Path(source).name}` (page: {page})")
                        st.write(doc.page_content)
                        st.markdown("---")
        except Exception as exc:
            st.error(f"Failed to generate answer: {exc}")
            if llm_provider == "openai":
                st.info(
                    "请检查 `.env` 中的 `OPENAI_API_KEY` 是否为真实的 `sk-` 开头密钥，"
                    "不能保留「你的key」这类中文占位符。"
                )
            elif llm_provider == "ollama":
                st.info(
                    "常见原因：① Ollama 未启动（运行 `ollama serve`）；"
                    "② 模型名填错或未下载。先执行 `ollama list` 查看已有模型，"
                    "或运行 `ollama pull gemma3:12b`。"
                )


if __name__ == "__main__":
    main()
