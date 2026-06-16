# Local Document QA System Based on RAG

## Project Overview

This project builds a local document question-answering system using Retrieval-Augmented Generation (RAG). Users can upload PDF or TXT documents and ask questions based on the document content.

## Tech Stack

- Python
- LangChain
- Chroma Vector Database
- Streamlit
- HuggingFace Embeddings (local)
- OpenAI API / Ollama (for answer generation)
- PDF document processing (PyPDF)

## Main Features

- Load and process local PDF/TXT documents
- Split long documents into text chunks
- Convert text chunks into embeddings
- Store embeddings in Chroma vector database
- Retrieve relevant document chunks based on user questions
- Generate answers with context from the document

## Project Structure

```
rag-document-qa/
├── app.py                  # Streamlit web app
├── requirements.txt
├── README.md
├── data/
│   ├── sample.txt          # Sample document for testing
│   └── sample.pdf          # Sample PDF version
├── screenshots/
└── src/
    ├── load_documents.py   # Document loading
    ├── build_vectorstore.py # Text splitting & vector storage
    └── qa_chain.py         # RAG QA chain
```

## How to Run

### 1. Install dependencies

```bash
cd rag-document-qa
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure LLM (choose one)

**Option A: OpenAI**

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="gpt-4o-mini"   # optional
```

**Option B: Ollama (local, free)**

```bash
ollama pull qwen2.5:3b
ollama serve
```

You can also create a `.env` file:

```
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-4o-mini
```

### 3. Run the app

```bash
streamlit run app.py
```

### 4. Use the app

1. Open the Streamlit page in your browser
2. Click **Build / Rebuild Knowledge Base** (uses `data/sample.txt` by default)
3. Enter a question, e.g. "What is RAG?" or "RAG 的流程是什么？"
4. Click **Get Answer**

## Sample Questions

- What is RAG?
- What are the steps of the RAG pipeline?
- Why do we need text chunking?
- How to reduce LLM hallucination?

## What I Learned

Through this project, I understood the basic workflow of RAG, including document loading, text splitting, embedding, vector retrieval, and answer generation. I also learned how to combine LangChain, Chroma, and Streamlit to build a simple but complete document QA system.

## Interview Notes

**RAG Pipeline:** Document Loading → Text Splitting → Embedding → Vector Storage → Similarity Retrieval → LLM Answer Generation

**Why RAG?** LLMs cannot access local/private documents directly. RAG retrieves relevant content first, then generates grounded answers.

**Optimization ideas:** Adjust chunk size, use better embeddings, add reranking, improve PDF parsing.
