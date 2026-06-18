"""
backend/rag_pipeline.py
=======================
Enterprise HR & IT Policy RAG Pipeline.

Responsibilities
----------------
Accepts a natural-language question, retrieves the most relevant document
chunks from a persisted FAISS index, assembles them into an attributed
context block, calls a local Ollama llama3.2 instance for grounded answer
generation, and returns a fully populated QueryResponse.

Public API
----------
Only ``answer_question()`` is intended for external use (e.g., from api.py).
All other functions are internal helpers or infrastructure and are prefixed
with an underscore where appropriate.

Performance
-----------
The HuggingFace embedding model and the FAISS vector store are lazy-loaded
on the first call and then cached as module-level singletons. A threading
lock prevents double-initialisation under concurrent requests.

Python 3.11+
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import requests
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from backend.config import (
    VECTOR_STORE_DIRECTORY,
    EMBEDDING_MODEL_NAME,
    TOP_K_RETRIEVAL,
)
from backend.models import QueryResponse, SourceMetadata

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ollama configuration — module-level constants, never inlined elsewhere
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = "http://localhost:11434"
OLLAMA_MODEL: str = "llama3.2"
OLLAMA_GENERATE_ENDPOINT: str = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_REQUEST_TIMEOUT_SECONDS: int = 120

# ---------------------------------------------------------------------------
# Hallucination-prevention fallback — single source of truth
# ---------------------------------------------------------------------------
FALLBACK_ANSWER: str = (
    "I could not find that information in the provided policy documents."
)

# ---------------------------------------------------------------------------
# Anti-hallucination prompt builder
#
# Using an f-string function instead of a str.format() template is deliberate.
# str.format() parses its *entire* input for {placeholder} tokens — so any
# retrieved policy chunk containing curly braces (JSON examples, regex
# patterns, Windows paths, etc.) raises KeyError and crashes the request.
# An f-string evaluates against local variables only; arbitrary text in
# `context` and `question` is treated as data, never as a format template.
# ---------------------------------------------------------------------------
def _build_prompt(context: str, question: str) -> str:
    """
    Construct the final LLM prompt from the assembled context and question.

    Combines the anti-hallucination instruction block, the retrieved context,
    and the user question into a single string ready for the Ollama API.
    Using an f-string function (rather than ``str.format()``) ensures that
    curly braces inside retrieved policy text are never mis-parsed as
    format placeholders.

    Parameters
    ----------
    context : str
        The assembled, attributed context block produced by ``_build_context()``.
    question : str
        The cleaned, stripped user question.

    Returns
    -------
    str
        The complete prompt string to send to Ollama.
    """
    return (
        "You are an Enterprise HR & IT Policy Assistant.\n\n"
        "STRICT INSTRUCTIONS — follow every rule without exception:\n"
        "1. Answer ONLY using the context provided between the delimiters below.\n"
        "2. Do NOT use your prior training knowledge under any circumstances.\n"
        "3. Do NOT make assumptions about information not explicitly stated.\n"
        "4. Do NOT infer or extrapolate missing information.\n"
        "5. Do NOT fabricate, hallucinate, or embellish any answer.\n\n"
        "If the answer to the question is not present in the context, you MUST respond\n"
        "with this exact sentence and nothing else:\n"
        f'"{FALLBACK_ANSWER}"\n\n'
        "--- CONTEXT START ---\n"
        f"{context}\n"
        "--- CONTEXT END ---\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

# ---------------------------------------------------------------------------
# Lazy-loaded module-level singletons
# ---------------------------------------------------------------------------
_VECTOR_STORE: Optional[FAISS] = None
_EMBEDDINGS: Optional[HuggingFaceEmbeddings] = None
_LOAD_LOCK: threading.Lock = threading.Lock()


# ---------------------------------------------------------------------------
# Infrastructure layer
# ---------------------------------------------------------------------------

def load_vector_store() -> FAISS:
    """
    Return the module-level FAISS vector store, loading it on first call.

    Implements thread-safe lazy-loading: the embedding model and FAISS index
    are initialised exactly once per process lifetime, regardless of concurrent
    callers. Subsequent calls return the cached singleton in O(1).

    Loading sequence
    ----------------
    1. Fast-path check: if ``_VECTOR_STORE`` is already set, return it
       immediately without acquiring the lock.
    2. Lock acquisition: prevents concurrent first-callers from double-loading.
    3. Double-checked guard inside the lock: re-check after acquiring in case
       another thread completed loading while this thread waited.
    4. Load ``HuggingFaceEmbeddings`` with the model name from config.
    5. Load ``FAISS.load_local`` from the configured directory.

    Returns
    -------
    FAISS
        The loaded and cached vector store instance.

    Raises
    ------
    FileNotFoundError
        If ``index.faiss`` does not exist at ``VECTOR_STORE_DIRECTORY``.
        Run the ingestion pipeline (``ingest.py``) first.
    RuntimeError
        If the embedding model or FAISS index fails to load.
    """
    global _VECTOR_STORE, _EMBEDDINGS

    # Fast path — no locking overhead after first load
    if _VECTOR_STORE is not None:
        logger.debug("Returning cached FAISS vector store.")
        return _VECTOR_STORE

    with _LOAD_LOCK:
        # Double-checked: another thread may have loaded while we waited
        if _VECTOR_STORE is not None:
            return _VECTOR_STORE

        vector_store_path = Path(VECTOR_STORE_DIRECTORY)
        index_file = vector_store_path / "index.faiss"

        if not index_file.exists():
            raise FileNotFoundError(
                f"FAISS index not found at '{index_file}'. "
                "Run the ingestion pipeline (ingest.py) to build the index first."
            )

        logger.info(
            "Loading embedding model '%s' (first load — this happens once).",
            EMBEDDING_MODEL_NAME,
        )
        try:
            _EMBEDDINGS = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialise embedding model '{EMBEDDING_MODEL_NAME}': {exc}"
            ) from exc

        logger.info(
            "Loading FAISS vector store from '%s'.",
            vector_store_path,
        )
        try:
            _VECTOR_STORE = FAISS.load_local(
                folder_path=str(vector_store_path),
                embeddings=_EMBEDDINGS,
                allow_dangerous_deserialization=True,
            )
        except Exception as exc:
            # Reset embeddings so a retry attempt can try again cleanly
            _EMBEDDINGS = None
            raise RuntimeError(
                f"Failed to load FAISS vector store from '{vector_store_path}': {exc}"
            ) from exc

        logger.info(
            "FAISS vector store loaded and cached. Embedding model: '%s'.",
            EMBEDDING_MODEL_NAME,
        )

    return _VECTOR_STORE


# ---------------------------------------------------------------------------
# Retrieval layer
# ---------------------------------------------------------------------------

def retrieve_documents(question: str) -> list[Document]:
    """
    Retrieve the top-K most relevant document chunks for a question.

    Embeds the question using the cached ``HuggingFaceEmbeddings`` model and
    performs a dense-vector similarity search against the cached FAISS index.
    This function is intentionally kept minimal: it retrieves and returns; it
    does not build context, call an LLM, or mutate state.

    Parameters
    ----------
    question : str
        The natural-language question from the end user. Must be non-empty.

    Returns
    -------
    list[Document]
        Up to ``TOP_K_RETRIEVAL`` LangChain ``Document`` objects, ordered by
        descending cosine similarity to the question embedding. Each document
        carries a ``metadata`` dict with at least ``"source"`` and ``"page"``
        keys (set by the ingestion pipeline).

    Raises
    ------
    ValueError
        If ``question`` is empty or whitespace-only.
    RuntimeError
        If FAISS similarity search fails for any reason.
    """
    if not question or not question.strip():
        raise ValueError("'question' must be a non-empty, non-whitespace string.")

    store: FAISS = load_vector_store()

    logger.info(
        "Performing similarity search | top_k=%d | question='%s'",
        TOP_K_RETRIEVAL,
        question[:150],
    )

    try:
        documents: list[Document] = store.similarity_search(
            query=question.strip(),
            k=TOP_K_RETRIEVAL,
        )
    except Exception as exc:
        raise RuntimeError(
            f"FAISS similarity search failed for question '{question[:60]}...': {exc}"
        ) from exc

    logger.info(
        "Retrieved %d/%d chunks from FAISS.",
        len(documents),
        TOP_K_RETRIEVAL,
    )
    return documents


# ---------------------------------------------------------------------------
# Context and citation assembly (private helpers)
# ---------------------------------------------------------------------------

def _build_context(documents: list[Document]) -> str:
    """
    Assemble retrieved chunks into a single attributed context block.

    Each chunk is prefixed with a structured header that identifies its
    ordinal position, source filename, and page number. The LLM can use
    these headers as grounding signals when constructing its answer.

    Format per chunk::

        [Chunk 1 | Source: it_security_policy.pdf | Page: 4]
        <chunk text>

    Chunks are separated by a blank line to preserve visual boundaries.

    Parameters
    ----------
    documents : list[Document]
        Retrieved LangChain ``Document`` objects in relevance-descending order.

    Returns
    -------
    str
        A single multi-line string ready for injection into the prompt template.
        Returns an empty string if ``documents`` is empty.
    """
    if not documents:
        return ""

    parts: list[str] = []

    for idx, doc in enumerate(documents, start=1):
        raw_source: str = doc.metadata.get("source", "Unknown Source")
        page: int | str = doc.metadata.get("page", "N/A")

        # Normalise to bare filename; avoid exposing server filesystem paths
        source_name: str = (
            Path(raw_source).name
            if raw_source != "Unknown Source"
            else raw_source
        )

        header = f"[Chunk {idx} | Source: {source_name} | Page: {page}]"
        body = doc.page_content.strip()
        parts.append(f"{header}\n{body}")

    return "\n\n".join(parts)


def _build_sources(documents: list[Document]) -> list[SourceMetadata]:
    """
    Extract deduplicated ``SourceMetadata`` citation objects from retrieved docs.

    When multiple retrieved chunks originate from the same (filename, page)
    pair, only a single ``SourceMetadata`` entry is produced — the final
    citations list contains one entry per unique document page, not per chunk.

    Parameters
    ----------
    documents : list[Document]
        Retrieved LangChain ``Document`` objects.

    Returns
    -------
    list[SourceMetadata]
        Deduplicated citation objects ordered by first appearance in the
        retrieval results.
    """
    seen: set[tuple[str, int | str]] = set()
    sources: list[SourceMetadata] = []

    for doc in documents:
        raw_source: str = doc.metadata.get("source", "Unknown Source")
        raw_page = doc.metadata.get("page", 0)

        source_name: str = (
            Path(raw_source).name
            if raw_source != "Unknown Source"
            else raw_source
        )

        # Safely coerce page to int. Some loaders (e.g. UnstructuredPDFLoader)
        # store page as a string. Pydantic v2 in strict mode will reject a
        # non-int value for `page: int`, so we normalise here rather than
        # relying on Pydantic's optional coercion.
        try:
            page = max(1, int(raw_page))
        except (ValueError, TypeError):
            logger.warning(
                "Could not coerce page value '%s' to int for source '%s'. "
                "Defaulting to 1.",
                raw_page,
                source_name,
            )
            page = 1

        dedup_key = (source_name, page)
        if dedup_key not in seen:
            seen.add(dedup_key)
            sources.append(SourceMetadata(source=source_name, page=page))

    return sources


# ---------------------------------------------------------------------------
# LLM layer (private)
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str) -> str:
    """
    Send a fully assembled prompt to the local Ollama instance and return
    the generated answer text.

    Uses the Ollama ``/api/generate`` endpoint with ``stream=False`` for
    deterministic, blocking response handling. ``temperature`` is set to
    ``0.0`` to eliminate stochastic generation and enforce policy-accurate,
    reproducible answers.

    Parameters
    ----------
    prompt : str
        The complete, assembled prompt string including the context block,
        instructions, and user question.

    Returns
    -------
    str
        The generated answer text from the model. If the model returns an
        empty string, ``FALLBACK_ANSWER`` is returned instead.

    Raises
    ------
    RuntimeError
        On connection failure, HTTP errors, or request timeout. The error
        message includes actionable context (e.g., whether Ollama is running).
    """
    payload: dict = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,   # Deterministic — critical for grounded RAG
            "top_p": 1.0,
            "num_predict": 1024,
        },
    }

    logger.info(
        "Calling Ollama | model=%s | endpoint=%s | prompt_length=%d chars",
        OLLAMA_MODEL,
        OLLAMA_GENERATE_ENDPOINT,
        len(prompt),
    )

    try:
        response = requests.post(
            url=OLLAMA_GENERATE_ENDPOINT,
            json=payload,
            timeout=OLLAMA_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()

    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"Ollama request timed out after {OLLAMA_REQUEST_TIMEOUT_SECONDS}s. "
            "Consider increasing OLLAMA_REQUEST_TIMEOUT_SECONDS or reducing context size."
        ) from exc

    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Cannot connect to Ollama at '{OLLAMA_BASE_URL}'. "
            "Ensure the Ollama server is running: `ollama serve`."
        ) from exc

    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(
            f"Ollama API returned HTTP {response.status_code}: {response.text[:300]}"
        ) from exc

    response_data: dict = response.json()
    answer: str = response_data.get("response", "").strip()

    if not answer:
        logger.warning(
            "Ollama returned an empty response for model '%s'. "
            "Returning fallback answer.",
            OLLAMA_MODEL,
        )
        return FALLBACK_ANSWER

    logger.info(
        "Ollama response received | model=%s | answer_length=%d chars",
        OLLAMA_MODEL,
        len(answer),
    )
    return answer


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def answer_question(question: str) -> QueryResponse:
    """
    Execute the end-to-end RAG pipeline and return a grounded answer.

    This is the **sole public entry point** for the RAG pipeline. It
    orchestrates all internal stages and is the only function that callers
    (e.g., ``api.py`` route handlers) should invoke.

    Pipeline stages
    ---------------
    1. **Validate** — reject empty or whitespace-only questions early.
    2. **Retrieve** — embed the question and fetch top-K chunks from FAISS.
    3. **Guard** — if no documents are retrieved, return the fallback answer
       immediately without calling the LLM (saves latency and cost).
    4. **Assemble context** — format retrieved chunks into an attributed block.
    5. **Build prompt** — inject context and question into the anti-hallucination
       prompt template.
    6. **Generate** — call Ollama llama3.2 with ``temperature=0.0``.
    7. **Extract citations** — deduplicate source metadata from chunk metadata.
    8. **Return** — produce a ``QueryResponse`` with answer and sources.

    Parameters
    ----------
    question : str
        The natural-language question from the end user. Must be a non-empty
        string.

    Returns
    -------
    QueryResponse
        Contains:
        - ``answer``: The grounded answer generated from retrieved context,
          or ``FALLBACK_ANSWER`` if no relevant context was found.
        - ``sources``: A deduplicated list of ``SourceMetadata`` objects
          identifying the policy documents and page numbers cited.

    Raises
    ------
    ValueError
        If ``question`` is empty or whitespace-only.
    RuntimeError
        If retrieval fails (FAISS error) or LLM generation fails (Ollama
        unreachable or HTTP error). Callers should catch this and return an
        appropriate HTTP 500 response.

    Examples
    --------
    >>> from backend.rag_pipeline import answer_question
    >>> result = answer_question("What is the remote work policy?")
    >>> print(result.answer)
    >>> for src in result.sources:
    ...     print(src.source, src.page)
    """
    if not question or not question.strip():
        raise ValueError("'question' must be a non-empty string.")

    clean_question: str = question.strip()

    logger.info(
        "RAG pipeline started | question='%s'",
        clean_question[:200],
    )

    # ── Stage 1: Retrieve relevant chunks ────────────────────────────────────
    documents: list[Document] = retrieve_documents(clean_question)

    # ── Stage 2: Guard — no context found ────────────────────────────────────
    if not documents:
        logger.warning(
            "No documents retrieved from FAISS for question='%s'. "
            "Returning fallback answer without calling Ollama.",
            clean_question[:150],
        )
        return QueryResponse(
            answer=FALLBACK_ANSWER,
            sources=[],
        )

    # ── Stage 3: Assemble context block ──────────────────────────────────────
    context: str = _build_context(documents)
    logger.debug(
        "Context assembled | chunks=%d | context_length=%d chars",
        len(documents),
        len(context),
    )

    # ── Stage 4: Build prompt ─────────────────────────────────────────────────
    prompt: str = _build_prompt(context=context, question=clean_question)

    # ── Stage 5: Call Ollama ──────────────────────────────────────────────────
    answer: str = _call_ollama(prompt)

    # ── Stage 6: Extract deduplicated citations ───────────────────────────────
    sources: list[SourceMetadata] = _build_sources(documents)

    logger.info(
        "RAG pipeline complete | answer_length=%d chars | citations=%d",
        len(answer),
        len(sources),
    )

    return QueryResponse(answer=answer, sources=sources)