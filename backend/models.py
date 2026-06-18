"""
backend/models.py
=================
Pydantic v2 schema models for the Enterprise HR & IT Policy RAG Assistant.

Scope
-----
This module owns the complete API contract for the /query endpoint:
  - QueryRequest  — validated incoming payload
  - SourceMetadata — provenance record for a single retrieved chunk
  - QueryResponse  — structured response envelope

Intentional boundaries
-----------------------
* Zero business logic.
* No I/O, no database access, no LLM calls.
* No FastAPI endpoint definitions.
* No inter-model dependencies beyond composition.

These models are the *single source of truth* for request/response shapes
consumed by FastAPI's automatic OpenAPI/Swagger documentation, as well as by
the Streamlit frontend through the shared contract.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# QueryRequest
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """
    Incoming payload for ``POST /query``.

    Carries the user's natural-language question to be routed through the
    RAG pipeline (vector retrieval → context assembly → LLM synthesis).

    Validation pipeline (applied in order)
    ----------------------------------------
    1. Leading/trailing whitespace is stripped from ``question`` automatically
       via ``str_strip_whitespace=True`` in model config.
    2. Pydantic enforces ``min_length=1`` and ``max_length=512`` on the
       stripped value.
    3. The explicit field validator fires last and emits a domain-specific
       error message if the stripped value is still empty — providing
       developer-friendly feedback over Pydantic's generic length error.

    Immutability
    ------------
    ``frozen=True`` prevents any in-flight mutation of the question after the
    model is constructed, making instances safe to pass across thread/async
    boundaries in FastAPI's request handling.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,  # strip all str fields before validation
        frozen=True,                # immutable after construction
    )

    question: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description=(
            "Natural-language question submitted by the user. "
            "Leading/trailing whitespace is trimmed automatically. "
            "Must be between 1 and 512 characters after trimming. "
            "The 512-character ceiling is a lightweight guard against "
            "oversized payloads being injected as prompt context."
        ),
        examples=["How many sick leaves are allowed per year?"],
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        """
        Reject questions that are empty after whitespace stripping.

        Although ``str_strip_whitespace=True`` combined with ``min_length=1``
        would already block this case, the explicit validator surfaces a
        domain-specific error message suited for structured API error responses
        and client-side display — rather than Pydantic's generic length error.

        Note: ``value`` is already stripped by the time this validator fires.
        """
        if not value:
            raise ValueError(
                "question must contain at least one non-whitespace character."
            )
        return value


# ---------------------------------------------------------------------------
# SourceMetadata
# ---------------------------------------------------------------------------


class SourceMetadata(BaseModel):
    """
    Provenance record for a single document chunk returned by the retriever.

    Purpose
    -------
    Every answer produced by the RAG pipeline must be traceable to its
    source material. This model captures the exact coordinates of a retrieved
    chunk so that:

    * End-users can navigate to and verify cited passages.
    * Compliance / audit workflows can confirm no fabricated content.
    * The frontend can render rich inline citations with document links.

    Field constraints
   -----------------
    source has min_length=1 to prevent empty filenames slipping through
    from inconsistent document-parsing output.

    page enforces ge=1 because all real document formats are 1-indexed;
    allowing page=0 would silently introduce off-by-one bugs. 

    ``str_strip_whitespace=True`` normalises filenames and section labels
    that may carry extraneous whitespace from PDF or DOCX metadata extraction.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,  # normalise metadata strings from parsers
        frozen=True,                # provenance records are read-only by nature
    )

    source: str = Field(
        ...,
        min_length=1,
        description=(
            "File name or unique identifier of the source document. "
            "Examples: 'HR_Policy.pdf', 'IT_Security_Guidelines_v2.docx'. "
            "Used by the frontend to construct document download / preview links."
        ),
        examples=["HR_Policy.pdf"],
    )
    
    page: int = Field(
        ...,
        ge=1,
        description=(
            "1-based page number in the source document where the "
            "retrieved evidence chunk appears. "
            "Minimum value is 1 (pages are never zero-indexed)."
        ),
        examples=[5],
    )


# ---------------------------------------------------------------------------
# QueryResponse
# ---------------------------------------------------------------------------


class QueryResponse(BaseModel):
    """
    Response envelope for ``POST /query``.

    Structure
    ---------
    Pairs the LLM-synthesised answer with its complete provenance trail.
    Keeping both fields in one typed model guarantees the frontend always
    receives citations alongside the answer — neither field can be omitted
    by accident.

    ``sources`` defaults to an empty list (not ``None``) so that consumers
    can always iterate over it without a null-check, even when the retriever
    found no relevant chunks (e.g. question outside corpus scope).

    Serialisation note
    ------------------
    ``str_strip_whitespace`` is intentionally *not* applied here.  The
    ``answer`` field is raw LLM output; stripping could remove meaningful
    leading/trailing whitespace in formatted prose or code blocks.

    ``frozen=True`` ensures the response object is immutable once constructed
    by the service layer, preventing any accidental mutation before
    FastAPI serialises it to JSON.
    """

    model_config = ConfigDict(
        frozen=True,  # response objects are constructed once, then serialised
    )

    answer: str = Field(
        ...,
        min_length=1,
        description=(
            "Answer synthesised by the LLM from the retrieved document chunks. "
            "Reflects only information grounded in the source corpus. "
            "Preserves original LLM formatting including newlines and spacing."
        ),
        examples=["Employees are entitled to 12 sick leaves per year."],
    )

    sources: List[SourceMetadata] = Field(
        default_factory=list,
        description=(
            "Ordered list of document chunks used to ground the answer, "
            "sorted by retrieval relevance score (highest first). "
            "An empty list indicates the LLM had no retrieved context to cite — "
            "for example, when the question falls outside the indexed corpus."
        ),
    )
