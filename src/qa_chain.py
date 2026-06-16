"""Create a RAG question-answering chain."""

import os
from typing import Any

from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the question based only on the context below.\n"
            "If the answer is not in the context, say you cannot find it in the document.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{input}"),
    ]
)


def _validate_openai_api_key(api_key: str) -> None:
    """Reject placeholder or non-ASCII API keys with a clear message."""
    placeholders = {"你的key", "your-api-key", "your_api_key", "sk-your-key", "xxx", "test"}
    if api_key.strip().lower() in placeholders or api_key.startswith("你的"):
        raise ValueError(
            "OPENAI_API_KEY 仍是占位符，请改成真实的 sk- 开头密钥。"
            "获取地址：https://platform.openai.com/api-keys"
        )
    try:
        api_key.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(
            "OPENAI_API_KEY 含有中文或特殊字符，HTTP 请求头无法编码。"
            "请填入 OpenAI 官方提供的 sk- 开头英文密钥。"
        ) from exc


def create_llm(provider: str, model_name: str | None = None) -> BaseLanguageModel:
    """Create an LLM from OpenAI or Ollama."""
    provider = provider.lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Add it to your environment or .env file.")
        _validate_openai_api_key(api_key)

        return ChatOpenAI(
            model=model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            api_key=api_key,
        )

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise ImportError(
                "Ollama 支持需要安装 langchain-ollama：pip install langchain-ollama"
            ) from exc

        return ChatOllama(
            model=model_name or os.getenv("OLLAMA_MODEL", "gemma3:12b"),
            temperature=0,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}. Use 'openai' or 'ollama'.")


def create_qa_chain(
    vectorstore: Chroma,
    llm: BaseLanguageModel,
    k: int = 4,
):
    """Build a retrieval-augmented QA chain."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    combine_docs_chain = create_stuff_documents_chain(llm, QA_PROMPT)
    return create_retrieval_chain(retriever, combine_docs_chain)


def ask_question(chain, question: str) -> dict[str, Any]:
    """Run a question through the QA chain."""
    result = chain.invoke({"input": question})
    return {
        "result": result.get("answer", ""),
        "source_documents": result.get("context", []),
    }
