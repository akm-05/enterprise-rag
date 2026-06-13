"""
backend/config.py
=================

Centralized application configuration for the Enterprise HR & IT Policy
RAG Assistant.

This file acts as the single source of truth for:
- Embedding model settings
- Document ingestion settings
- Vector database settings
- Retrieval settings

Keeping these values here avoids hardcoding configuration across multiple files.
"""

# ---------------------------------------------------------------------------
# Embedding Configuration
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# ---------------------------------------------------------------------------
# Document Ingestion Configuration
# ---------------------------------------------------------------------------

DATA_DIRECTORY = "data"

CHUNK_SIZE = 800

CHUNK_OVERLAP = 200

# ---------------------------------------------------------------------------
# Vector Store Configuration
# ---------------------------------------------------------------------------

VECTOR_STORE_DIRECTORY = "vector_store"

# ---------------------------------------------------------------------------
# Retrieval Configuration
# ---------------------------------------------------------------------------

TOP_K_RETRIEVAL = 5