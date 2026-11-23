# DocuChat — Document Q&A with Retrieval-Augmented Generation (RAG)

A hands-on learning project implementing an end-to-end **Retrieval-Augmented Generation (RAG)** pipeline in Python using **LangChain**, **ChromaDB**, **HuggingFace Embeddings**, and **Google Gemini (`gemini-3.5-flash-lite`)**.

DocuChat loads a research PDF (*"Attention Is All You Need"*), splits it into overlapping chunks, indexes dense vector embeddings locally, and answers user queries strictly grounded in the retrieved document context.

---

## Architecture & Pipeline

The notebook (`DocuChat.ipynb`) is structured into two main stages:

### 1. Indexing Pipeline
1. **Load (`PyPDFLoader`):** Fetches and parses the source PDF directly from arXiv (`https://arxiv.org/pdf/1706.03762`) into LangChain `Document` objects with page-level metadata.
2. **Split (`RecursiveCharacterTextSplitter`):** Breaks the document into chunks of `1000` characters with a `200`-character overlap to preserve semantic continuity across chunk boundaries.
3. **Embed (`HuggingFaceEmbeddings`):** Generates dense vector representations using the `sentence-transformers/all-mpnet-base-v2` model.
4. **Store (`Chroma`):** Persists the embedded document chunks into a local Chroma vector database (`./chroma_langchain_db`, collection: `research_collection`).

### 2. Retrieve & Generate Pipeline
1. **Retrieve (`retrieve_context`):** Runs a top-$k$ similarity search (`k=2`) against the Chroma vector store for a given user query, returning both the formatted context string (`Source` metadata + `Content`) and the raw `Document` objects.
2. **Generate (`docu_chat`):** Injects the retrieved context into a strict system prompt instructing **`gemini-3.5-flash-lite`** (via `langchain-google-genai`) to answer step-by-step in bullet points using **only** the provided context without hallucinating external facts.

---

## Tech Stack

* **Orchestration:** LangChain (`langchain`, `langchain-community`, `langchain-text-splitters`)
* **LLM:** Google Gemini (`gemini-3.5-flash-lite` via `langchain-google-genai`)
* **Embeddings:** `sentence-transformers/all-mpnet-base-v2` (`langchain-huggingface`)
* **Vector Database:** ChromaDB (`langchain-chroma`)
* **Document Parser:** `PyPDF`
* **Environment:** Google Colab

---

## Setup & Usage

### 1. Install Dependencies
```bash
pip install -qU langchain langchain-community langchain-text-splitters langchain-huggingface sentence-transformers langchain-chroma langchain-google-genai pypdf
