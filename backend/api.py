"""
backend/api.py
==============
FastAPI application entry point for the Enterprise HR & IT Policy RAG Assistant.

This module bootstraps the FastAPI application, configures application-level
metadata, registers all routers, and defines the lifespan context for
startup/shutdown hooks required by future RAG pipeline components
(vector store loading, LLM initialization, etc.).

Architecture:
    Streamlit Frontend
        ↓
    FastAPI Backend  ← this module
        ↓
    RAG Pipeline
        ↓
    FAISS Vector Store
        ↓
    Local LLM (Llama / Mistral)
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import APIRouter, FastAPI
from pydantic import BaseModel

from backend.models import QueryRequest, QueryResponse
from backend.rag_pipeline import answer_question
from fastapi import HTTPException


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Schema returned by the health-check endpoint.

    Using a typed model (rather than a bare dict) ensures the response is
    validated on the way out and appears correctly in the OpenAPI schema.
    """

    status: str

    model_config = {
        "json_schema_extra": {
            "examples": [{"status": "healthy"}]
        }
    }


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown lifecycle.

    Startup:
        Future: load FAISS index, warm up the local LLM, etc.

    Shutdown:
        Future: flush caches, close DB connections, release GPU memory, etc.

    Using the lifespan pattern (introduced in FastAPI 0.93) instead of the
    deprecated @app.on_event("startup") / @app.on_event("shutdown") decorators
    keeps the codebase aligned with current FastAPI best practices.
    """
    # -- Startup -------------------------------------------------------
    # TODO: initialise vector store, LLM model, and connection pools here.
    yield
    # -- Shutdown ------------------------------------------------------
    # TODO: release resources (GPU memory, open file handles, etc.) here.


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


app = FastAPI(
    title="Enterprise HR & IT Policy RAG Assistant",
    description=(
        "A Retrieval-Augmented Generation (RAG) API that answers employee "
        "questions about HR and IT policies using a local LLM backed by a "
        "FAISS vector store.  All inference runs on-premises — no data leaves "
        "the corporate network."
    ),
    version="1.0.0",
    lifespan=lifespan,
    # Expose interactive docs only in non-production environments.
    # Override via environment variable in your deployment config.
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------


health_router = APIRouter(tags=["Health"])


@health_router.get(
    "/health",
    response_model=HealthResponse,
    status_code=200,
    summary="Health check",
    description=(
        "Returns the current operational status of the API server. "
        "Intended for use by load-balancers, container orchestrators (e.g. "
        "Kubernetes liveness/readiness probes), and uptime monitors."
    ),
)
async def health_check() -> HealthResponse:
    """Confirm the API process is running and able to serve requests."""
    return HealthResponse(status="healthy")


# Register routers
# Future routers (query, ingest, admin) are mounted here in the same pattern.
app.include_router(health_router)

# QUERY ROUTER
query_router = APIRouter(tags=["Query"])

@query_router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question",
)
async def query(request: QueryRequest) -> QueryResponse:
    try:
        return answer_question(request.question)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )

app.include_router(query_router)