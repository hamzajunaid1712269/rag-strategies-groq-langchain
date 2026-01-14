
"""
RAG Strategies (Basic RAG, RRR-RAG, Self-RAG) using:
- Groq (LLM)
- LangChain (Documents, Splitters)
- HuggingFaceEmbeddings (sentence-transformers/all-MiniLM-L6-v2)
- Chroma (local vector store)
- Optional web corpus indexing via WebBaseLoader (Lilian Weng posts)

This script is extracted/cleaned from the course notebook:
EngGenAI_Hamza_Junaid.ipynb


from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import bs4
from groq import Groq
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader


# ----------------------------
# Config
# ----------------------------

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"

DEFAULT_WEB_URLS = (
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
)


def _require_env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


def get_groq_client() -> Groq:
    return Groq(api_key=_require_env("GROQ_API_KEY"))


def call_groq_llm(prompt: str, model: str = DEFAULT_GROQ_MODEL) -> str:
    """
    Minimal Groq call, consistent with the notebook approach.
    """
    client = get_groq_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def build_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    # Normalize embeddings to improve cosine similarity behavior in practice.
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(
    docs: Sequence[Document],
    embeddings: HuggingFaceEmbeddings,
    collection_name: str,
    persist_dir: Optional[str] = None,
) -> Chroma:
    """
    Creates a Chroma vector store from documents.
    If persist_dir is provided, the index will be persisted locally.
    """
    return Chroma.from_documents(
        documents=list(docs),
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )


def split_docs(
    docs: Sequence[Document],
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(list(docs))


# ----------------------------
# Local corpus (from notebook)
# ----------------------------

def build_local_rag_docs() -> List[Document]:
    """
    A small explanatory corpus similar to the notebook's local RAG docs.
    """
    return [
        Document(page_content=(
            "Retrieval-Augmented Generation (RAG) combines information retrieval with a language model. "
            "Instead of relying only on the model’s internal parameters, RAG retrieves relevant external text "
            "and injects it into the prompt. This grounding step typically reduces hallucinations and improves "
            "factual accuracy because the model can base its answer on retrieved evidence."
        )),
        Document(page_content=(
            "A standard RAG pipeline includes: ingestion of source documents, chunking the text into overlapping "
            "segments, embedding each chunk into a dense vector representation, indexing those vectors in a "
            "vector database, retrieving top-k relevant chunks for a query, and generating an answer constrained "
            "to the retrieved context."
        )),
        Document(page_content=(
            "Chunking trades off context completeness and retrieval precision. Smaller chunks can improve precision "
            "but may lose cross-sentence context; overlap helps preserve continuity. Typical chunk sizes range from "
            "200–1,000 tokens depending on the domain and model context window."
        )),
        Document(page_content=(
            "Dense embeddings enable semantic retrieval by mapping text to vectors capturing meaning beyond exact "
            "keywords. Similarity search returns nearest neighbors to the query embedding, supporting paraphrases "
            "and conceptually related matches."
        )),
        Document(page_content=(
            "A refinement loop can improve RAG by rewriting the user query into a retrieval-friendly form, retrieving "
            "more relevant evidence, and then generating a better grounded answer. This is helpful when user questions "
            "are ambiguous, underspecified, or contain irrelevant wording."
        )),
    ]


# ----------------------------
# RAG strategies
# ----------------------------

def _make_context(docs: Sequence[Document]) -> str:
    return "\n\n".join([f"[Chunk {i+1}]\n{d.page_content.strip()}" for i, d in enumerate(docs)])


def basic_rag_answer(
    question: str,
    vectorstore: Chroma,
    k: int = 5,
    groq_model: str = DEFAULT_GROQ_MODEL,
) -> str:
    retrieved_docs = vectorstore.similarity_search(question, k=k)
    context = _make_context(retrieved_docs)

    prompt = f"""
You are a factual assistant. Answer the user's question using ONLY the context below.

Rules:
1) Use ONLY information explicitly stated in the context. Do not use outside knowledge.
2) If the context does not contain enough information to answer, reply exactly:
   "I don't know based on the provided context."
3) Do not guess, speculate, or invent details.
4) Keep the answer concise and grounded in the context.

Context:
{context}

Question:
{question}

Answer:
""".strip()

    return call_groq_llm(prompt, model=groq_model)


def rrr_rag_answer(
    question: str,
    vectorstore: Chroma,
    k: int = 5,
    groq_model: str = DEFAULT_GROQ_MODEL,
) -> str:
    """
    RRR-RAG: Rewrite → Retrieve → Respond
    """
    rewrite_prompt = f"""
You are a search query rewriter for a RAG system.
Rewrite the user's question into a concise, retrieval-friendly query.

Rules:
- Keep the meaning the same.
- Add key terms and expand acronyms if helpful.
- Remove fluff and conversational wording.
- Output ONLY the rewritten query (no quotes, no extra text).

Original question:
{question}

Rewritten query:
""".strip()

    rewritten_query = call_groq_llm(rewrite_prompt, model=groq_model).strip()
    docs = vectorstore.similarity_search(rewritten_query, k=k)
    ctx = _make_context(docs)

    answer_prompt = f"""
You are a factual assistant. Answer the question using ONLY the provided context.

Rules:
1) Use ONLY information explicitly stated in the context. Do not use outside knowledge.
2) If the context does not contain enough information to answer, reply exactly:
   "I don't know based on the provided context."
3) Do not guess, speculate, or invent details.
4) Keep the answer concise and grounded in the context.

Context:
{ctx}

Original Question:
{question}

Answer:
""".strip()

    return call_groq_llm(answer_prompt, model=groq_model)


def self_rag(
    question: str,
    vectorstore: Chroma,
    iterations: int = 2,
    k: int = 5,
    groq_model: str = DEFAULT_GROQ_MODEL,
) -> str:
    """
    Self-RAG: iterative retrieve → answer → refine query → repeat.
    """
    current_query = question
    answer = ""

    for i in range(iterations):
        docs = vectorstore.similarity_search(current_query, k=k)
        ctx = _make_context(docs)

        answer_prompt = f"""
You are a factual assistant. Answer the user's question using ONLY the provided context.

Rules:
1) Use ONLY information explicitly stated in the context. Do not use outside knowledge.
2) If the context does not contain enough information to answer, reply exactly:
   "I don't know based on the provided context."
3) Do not guess, speculate, or invent details.
4) Keep the answer concise and grounded in the context.

Iteration: {i+1} / {iterations}

Context:
{ctx}

Original Question:
{question}

Answer:
""".strip()

        answer = call_groq_llm(answer_prompt, model=groq_model).strip()

        refine_prompt = f"""
You are a search query optimizer for a RAG system.
Given the original question, the current retrieval query, and the model's latest answer,
produce a refined retrieval query to fetch better supporting evidence next iteration.

Rules:
- Keep the refined query short and retrieval-oriented.
- Preserve meaning, remove fluff, add missing key terms.
- Output ONLY the refined query.

Original question:
{question}

Current retrieval query:
{current_query}

Latest answer:
{answer}

Refined retrieval query:
""".strip()

        current_query = call_groq_llm(refine_prompt, model=groq_model).strip()

    return answer


# ----------------------------
# Web corpus indexing (from notebook)
# ----------------------------

def load_web_docs(urls: Sequence[str]) -> List[Document]:
    loader = WebBaseLoader(
        web_paths=tuple(urls),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
        ),
    )
    docs = loader.load()
    if not docs:
        raise RuntimeError("Web loader returned no documents. Check URLs/network access.")
    return docs


def build_web_vectorstore(
    urls: Sequence[str] = DEFAULT_WEB_URLS,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = 700,
    chunk_overlap: int = 100,
    persist_dir: Optional[str] = None,
) -> Chroma:
    web_docs = load_web_docs(urls)
    splits = split_docs(web_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embeddings = build_embeddings(embedding_model)
    return build_vectorstore(
        splits,
        embeddings,
        collection_name="web_corpus",
        persist_dir=persist_dir,
    )


# ----------------------------
# CLI
# ----------------------------

@dataclass
class Stores:
    local: Chroma
    web: Optional[Chroma] = None


def init_local_store(persist_dir: Optional[str], embedding_model: str) -> Chroma:
    docs = build_local_rag_docs()
    splits = split_docs(docs, chunk_size=300, chunk_overlap=50)
    embeddings = build_embeddings(embedding_model)
    return build_vectorstore(
        splits,
        embeddings,
        collection_name="local_rag_docs",
        persist_dir=persist_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG strategies demo (Groq + LangChain + Chroma).")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--groq-model", default=DEFAULT_GROQ_MODEL)
    parser.add_argument("--persist-dir", default=None, help="Optional directory for Chroma persistence.")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build_web = sub.add_parser("build-web", help="Build (and optionally persist) the web vector store.")
    p_build_web.add_argument("--urls", nargs="*", default=list(DEFAULT_WEB_URLS))
    p_build_web.add_argument("--chunk-size", type=int, default=700)
    p_build_web.add_argument("--chunk-overlap", type=int, default=100)

    p_ask = sub.add_parser("ask", help="Ask a question with a chosen RAG strategy.")
    p_ask.add_argument("--corpus", choices=["local", "web"], default="local")
    p_ask.add_argument("--strategy", choices=["basic", "rrr", "self"], default="basic")
    p_ask.add_argument("--question", required=True)
    p_ask.add_argument("--k", type=int, default=5)
    p_ask.add_argument("--iterations", type=int, default=2)

    args = parser.parse_args()

    # Initialize local store always (fast, small)
    local_store = init_local_store(args.persist_dir, args.embedding_model)

    if args.cmd == "build-web":
        _require_env("GROQ_API_KEY")  # not strictly needed for indexing, but keeps setup consistent
        web_store = build_web_vectorstore(
            urls=args.urls,
            embedding_model=args.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            persist_dir=args.persist_dir,
        )
        # trigger persistence if persist_dir is used
        if args.persist_dir:
            web_store.persist()
        print("[OK] Web vector store built.")
        return

    # ask
    store = local_store
    if args.corpus == "web":
        store = build_web_vectorstore(
            urls=DEFAULT_WEB_URLS,
            embedding_model=args.embedding_model,
            persist_dir=args.persist_dir,
        )

    if args.strategy == "basic":
        out = basic_rag_answer(args.question, store, k=args.k, groq_model=args.groq_model)
    elif args.strategy == "rrr":
        out = rrr_rag_answer(args.question, store, k=args.k, groq_model=args.groq_model)
    else:
        out = self_rag(args.question, store, iterations=args.iterations, k=args.k, groq_model=args.groq_model)

    print(out)


if __name__ == "__main__":
    main()
