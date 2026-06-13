"""
backend/ingest.py
=================
Document ingestion pipeline for the Enterprise HR & IT Policy RAG Assistant.

This module is the sole owner of the ingestion responsibility. It transforms
raw PDF policy documents stored in DATA_DIRECTORY into a persisted FAISS
vector store that the retrieval layer can query at runtime.

Pipeline stages (in execution order):
    1. Validate  — confirm DATA_DIRECTORY exists and is a directory
    2. Discover  — recursively locate all SUPPORTED_FILE_TYPES
    3. Load      — parse each PDF via LangChain PyPDFLoader
    4. Normalise — standardise metadata for portable, human-readable citations
    5. Split     — chunk text with RecursiveCharacterTextSplitter
    6. Embed     — generate dense vectors via HuggingFace sentence-transformer
    7. Index     — construct in-memory FAISS vector store
    8. Persist   — write index + docstore to VECTOR_STORE_DIRECTORY

Public API:
    create_vector_store() — runs the complete pipeline end-to-end

Out of scope (intentionally absent):
    - Retrieval / similarity search
    - Query answering / LLM inference
    - FastAPI endpoints
    - Streamlit UI
    - Hybrid retrieval or reranking
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Config import — compatible with both module and direct execution contexts.
#
#   python -m backend.ingest     →  `from backend.config import …`
#   python  backend/ingest.py    →  `from config import …`
# ---------------------------------------------------------------------------
try:
    from backend.config import (
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        DATA_DIRECTORY,
        EMBEDDING_MODEL_NAME,
        SUPPORTED_FILE_TYPES,
        VECTOR_STORE_DIRECTORY,
    )
except ImportError:
    from config import (  # type: ignore[no-redef]
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        DATA_DIRECTORY,
        EMBEDDING_MODEL_NAME,
        SUPPORTED_FILE_TYPES,
        VECTOR_STORE_DIRECTORY,
    )

# ---------------------------------------------------------------------------
# Module-level logger — callers configure the root logger; we never call
# basicConfig() here so we don't hijack the host application's log setup.
# ---------------------------------------------------------------------------
logger: Final[logging.Logger] = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------
__all__: list[str] = ["create_vector_store", "IngestionError",
                      "DataDirectoryNotFoundError", "NoDocumentsFoundError"]


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class IngestionError(Exception):
    """Base class for all ingestion pipeline failures.

    Catching this type lets callers handle any ingestion problem without
    needing to import every subclass.
    """


class DataDirectoryNotFoundError(IngestionError):
    """Raised when DATA_DIRECTORY does not exist or is not a directory."""


class NoDocumentsFoundError(IngestionError):
    """Raised when no supported documents can be loaded from DATA_DIRECTORY."""


# ---------------------------------------------------------------------------
# Stage 1 — Validate
# ---------------------------------------------------------------------------


def _validate_data_directory(data_dir: Path) -> None:
    """Assert that *data_dir* is a readable directory.

    Args:
        data_dir: Resolved path to the document repository.

    Raises:
        DataDirectoryNotFoundError: Path does not exist or is not a directory.
    """
    logger.info("Stage 1 | Validating data directory: '%s'", data_dir)

    if not data_dir.exists():
        raise DataDirectoryNotFoundError(
            f"DATA_DIRECTORY '{data_dir}' does not exist. "
            "Create the directory and place PDF policy documents inside it before "
            "running ingestion."
        )

    if not data_dir.is_dir():
        raise DataDirectoryNotFoundError(
            f"DATA_DIRECTORY '{data_dir}' exists but is not a directory. "
            "Check your config.py settings."
        )

    logger.info("Stage 1 | Data directory validated successfully.")


# ---------------------------------------------------------------------------
# Stage 2 — Discover
# ---------------------------------------------------------------------------


def _discover_pdf_files(data_dir: Path) -> list[Path]:
    """Recursively locate all files matching SUPPORTED_FILE_TYPES.

    Traversal is recursive so policy documents can be organised in
    subdirectories (e.g. data/hr/, data/it/) without any config changes.

    Args:
        data_dir: Validated path to the document repository.

    Returns:
        Sorted list of absolute Paths for each discovered file.

    Raises:
        NoDocumentsFoundError: Directory contains no supported files.
    """
    logger.info(
        "Stage 2 | Discovering %s files in: '%s'",
        sorted(SUPPORTED_FILE_TYPES),
        data_dir,
    )

    discovered: list[Path] = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_FILE_TYPES
    )

    if not discovered:
        raise NoDocumentsFoundError(
            f"No files with extensions {sorted(SUPPORTED_FILE_TYPES)} found "
            f"in '{data_dir}'. Add PDF policy documents and re-run ingestion."
        )

    logger.info("Stage 2 | Discovered %d file(s).", len(discovered))
    for path in discovered:
        logger.debug("Stage 2 |   → %s", path.relative_to(data_dir))

    return discovered


# ---------------------------------------------------------------------------
# Stage 3 — Load
# ---------------------------------------------------------------------------


def _load_documents(pdf_files: list[Path]) -> list[Document]:
    """Parse each PDF into LangChain Documents with per-file error isolation.

    A single corrupted or image-only PDF does NOT abort the pipeline. It is
    skipped with a WARNING so that all remaining valid files are processed.
    Only if every file fails does the function raise.

    Args:
        pdf_files: Discovered file paths from _discover_pdf_files.

    Returns:
        Flat list of Document objects — one Document per PDF page.

    Raises:
        NoDocumentsFoundError: Every file failed or yielded no extractable text.
    """
    logger.info("Stage 3 | Loading %d file(s).", len(pdf_files))

    loaded_documents: list[Document] = []
    failed_files: list[str] = []

    for pdf_path in pdf_files:
        logger.info("Stage 3 | Loading: '%s'", pdf_path.name)

        # --- attempt to parse ------------------------------------------------
        try:
            loader = PyPDFLoader(str(pdf_path))
            raw_pages: list[Document] = loader.load()
        except Exception as exc:
            logger.warning(
                "Stage 3 | Could not load '%s' — skipping. Reason: %s",
                pdf_path.name,
                exc,
            )
            failed_files.append(pdf_path.name)
            continue

        # --- filter blank / image-only pages ---------------------------------
        pages_with_content = [
            doc for doc in raw_pages if doc.page_content.strip()
        ]
        blank_page_count = len(raw_pages) - len(pages_with_content)

        if not pages_with_content:
            logger.warning(
                "Stage 3 | '%s' produced no extractable text (possibly scanned "
                "or image-only). Skipping.",
                pdf_path.name,
            )
            failed_files.append(pdf_path.name)
            continue

        if blank_page_count:
            logger.debug(
                "Stage 3 | '%s': %d blank/image-only page(s) omitted.",
                pdf_path.name,
                blank_page_count,
            )

        logger.info(
            "Stage 3 | '%s' loaded: %d page(s) with extractable content.",
            pdf_path.name,
            len(pages_with_content),
        )
        loaded_documents.extend(pages_with_content)

    # --- post-load summary ---------------------------------------------------
    if failed_files:
        logger.warning(
            "Stage 3 | %d file(s) skipped due to errors: %s",
            len(failed_files),
            ", ".join(failed_files),
        )

    if not loaded_documents:
        raise NoDocumentsFoundError(
            "All discovered files failed to load or contained no extractable "
            "text. Ensure the PDFs are valid and text-based (not scanned images)."
        )

    logger.info(
        "Stage 3 | Total pages loaded: %d across %d file(s).",
        len(loaded_documents),
        len(pdf_files) - len(failed_files),
    )
    return loaded_documents


# ---------------------------------------------------------------------------
# Stage 4 — Normalise metadata
# ---------------------------------------------------------------------------


def _normalise_metadata(documents: list[Document]) -> list[Document]:
    """Standardise metadata on every Document for reliable citation at retrieval.

    LangChain's PyPDFLoader produces:
        - ``source``: absolute file path on disk  →  machine-specific, fragile
        - ``page``:   0-based integer             →  unnatural for users

    This stage normalises to:
        - ``source``: filename only (e.g. ``it_security_policy.pdf``)
                      → portable across machines, safe in API responses
        - ``page``:   1-based integer             →  matches PDF viewer page numbers

    The normalised fields are the canonical citation identifiers used by the
    retrieval layer when it returns source attribution to the frontend.

    Args:
        documents: Raw Documents from _load_documents.

    Returns:
        The same list with metadata mutated in-place (avoids copying large text).
    """
    logger.info(
        "Stage 4 | Normalising metadata on %d document(s).", len(documents)
    )

    for doc in documents:
        raw_source: str = doc.metadata.get("source", "")
        raw_page: int = doc.metadata.get("page", 0)

        doc.metadata["source"] = Path(raw_source).name   # filename only
        doc.metadata["page"] = raw_page + 1              # 0-based → 1-based

    logger.info("Stage 4 | Metadata normalised.")
    return documents


# ---------------------------------------------------------------------------
# Stage 5 — Split
# ---------------------------------------------------------------------------


def _split_documents(documents: list[Document]) -> list[Document]:
    """Chunk documents using RecursiveCharacterTextSplitter.

    Separators are ordered from coarsest to finest so the splitter prefers
    semantic boundaries (paragraph → sentence → word → character) before
    making hard cuts.

    Chunk parameters are read from config.py so they can be tuned for
    different policy corpus characteristics without touching this file.

    Args:
        documents: Normalised list of loaded Documents.

    Returns:
        Flat list of text chunks. Each chunk inherits the source Document's
        metadata (source filename + page number).
    """
    logger.info(
        "Stage 5 | Splitting %d document(s) — "
        "chunk_size=%d, chunk_overlap=%d.",
        len(documents),
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = splitter.split_documents(documents)

    if not chunks:
        raise IngestionError(
            "Text splitting produced zero chunks. "
            "Check CHUNK_SIZE in config.py — it may be too large for the corpus."
        )

    logger.info(
        "Stage 5 | Produced %d chunk(s) from %d document(s).",
        len(chunks),
        len(documents),
    )
    return chunks


# ---------------------------------------------------------------------------
# Stage 6 — Embed
# ---------------------------------------------------------------------------


def _build_embeddings() -> HuggingFaceEmbeddings:
    """Initialise the HuggingFace sentence-transformer embedding model.

    The model is downloaded from HuggingFace Hub on first use and cached
    locally by the sentence-transformers library (typically under
    ~/.cache/huggingface/). Subsequent runs use the local cache.

    ``normalize_embeddings=True`` produces unit-length vectors so that inner
    product and cosine similarity are equivalent — important for FAISS
    IndexFlatIP searches.

    Returns:
        A configured HuggingFaceEmbeddings instance ready for inference.
    """
    logger.info(
        "Stage 6 | Initialising embedding model: '%s'", EMBEDDING_MODEL_NAME
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},          # swap to "cuda" when a GPU is available
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info("Stage 6 | Embedding model ready.")
    return embeddings


# ---------------------------------------------------------------------------
# Stage 7 — Index
# ---------------------------------------------------------------------------


def _build_faiss_index(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings,
) -> FAISS:
    """Embed all chunks and construct an in-memory FAISS vector store.

    ``FAISS.from_documents`` handles batching internally. All chunk metadata
    (source filename, page number) is stored inside the FAISS docstore and
    will be returned alongside retrieved chunks at query time.

    Args:
        chunks:     Text chunks from _split_documents.
        embeddings: Initialised embedding model from _build_embeddings.

    Returns:
        Populated in-memory FAISS vector store.
    """
    logger.info(
        "Stage 7 | Building FAISS index from %d chunk(s).", len(chunks)
    )

    vector_store: FAISS = FAISS.from_documents(chunks, embeddings)

    logger.info("Stage 7 | FAISS index built successfully.")
    return vector_store


# ---------------------------------------------------------------------------
# Stage 8 — Persist
# ---------------------------------------------------------------------------


def _persist_vector_store(vector_store: FAISS, store_dir: Path) -> None:
    """Write the FAISS index and docstore to disk.

    Creates *store_dir* (and any missing parent directories) if absent.

    Written files:
        ``index.faiss`` — binary FAISS index (the raw embedding vectors)
        ``index.pkl``   — pickled docstore (text chunks + metadata)

    Both files must be present for the retrieval layer to load the store.

    Args:
        vector_store: Populated FAISS instance from _build_faiss_index.
        store_dir:    Target directory defined by VECTOR_STORE_DIRECTORY.
    """
    logger.info("Stage 8 | Persisting vector store to: '%s'", store_dir)

    store_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(store_dir))

    written_files = [p.name for p in store_dir.iterdir()]
    logger.info(
        "Stage 8 | Vector store persisted. Files: %s", written_files
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_vector_store() -> None:
    """Run the complete document ingestion pipeline.

    Orchestrates all eight pipeline stages in sequence. This is the only
    function intended for use by external callers (FastAPI startup events,
    CLI scripts, test fixtures, etc.).

    Stages:
        1. Validate  — DATA_DIRECTORY exists and is readable
        2. Discover  — locate all SUPPORTED_FILE_TYPES recursively
        3. Load      — parse each PDF; skip corrupted / blank files gracefully
        4. Normalise — portable filenames + 1-based page numbers for citations
        5. Split     — chunk text via RecursiveCharacterTextSplitter
        6. Embed     — generate dense vectors with HuggingFace model
        7. Index     — build in-memory FAISS vector store
        8. Persist   — write index to VECTOR_STORE_DIRECTORY

    Raises:
        DataDirectoryNotFoundError: DATA_DIRECTORY is missing.
        NoDocumentsFoundError:      No usable PDFs found or all failed to load.
        IngestionError:             Any other unrecoverable pipeline failure.

    Example::

        from backend.ingest import create_vector_store
        create_vector_store()
    """
    _SEPARATOR = "=" * 60
    logger.info(_SEPARATOR)
    logger.info("Ingestion pipeline starting.")
    logger.info("  DATA_DIRECTORY      : %s", DATA_DIRECTORY)
    logger.info("  VECTOR_STORE_DIR    : %s", VECTOR_STORE_DIRECTORY)
    logger.info("  EMBEDDING_MODEL     : %s", EMBEDDING_MODEL_NAME)
    logger.info("  CHUNK_SIZE          : %d", CHUNK_SIZE)
    logger.info("  CHUNK_OVERLAP       : %d", CHUNK_OVERLAP)
    logger.info("  SUPPORTED_TYPES     : %s", sorted(SUPPORTED_FILE_TYPES))
    logger.info(_SEPARATOR)

    data_dir = Path(DATA_DIRECTORY).resolve()
    store_dir = Path(VECTOR_STORE_DIRECTORY).resolve()

    try:
        # Stage 1 — validate
        _validate_data_directory(data_dir)

        # Stage 2 — discover
        pdf_files = _discover_pdf_files(data_dir)

        # Stage 3 — load (per-file errors are isolated inside)
        documents = _load_documents(pdf_files)

        # Stage 4 — normalise metadata
        documents = _normalise_metadata(documents)

        # Stage 5 — split
        chunks = _split_documents(documents)

        # Stage 6 — embed
        embeddings = _build_embeddings()

        # Stage 7 — index
        vector_store = _build_faiss_index(chunks, embeddings)

        # Stage 8 — persist
        _persist_vector_store(vector_store, store_dir)

    except (DataDirectoryNotFoundError, NoDocumentsFoundError):
        # Domain exceptions are already logged at origin; re-raise cleanly.
        raise

    except IngestionError:
        # IngestionError subclasses from stage 5+ are also re-raised as-is.
        raise

    except Exception as exc:
        logger.exception(
            "Unexpected error in ingestion pipeline: %s", exc
        )
        raise IngestionError(
            f"Ingestion pipeline encountered an unexpected error: {exc}"
        ) from exc

    logger.info(_SEPARATOR)
    logger.info("Ingestion pipeline completed successfully.")
    logger.info(_SEPARATOR)


# ---------------------------------------------------------------------------
# Entry point — run directly or as a module:
#
#   python -m backend.ingest       (recommended — resolves imports correctly)
#   python   backend/ingest.py     (also works via try/except import block)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    create_vector_store()