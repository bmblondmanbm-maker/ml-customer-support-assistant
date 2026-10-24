"""Structured output schemas for the assistant."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class IntentResult(BaseModel):
    """Structured intent-classification result."""

    intent_id: Optional[int] = Field(default=None, ge=0)

    intent_name: Optional[str] = None

    confidence: float = Field(ge=0.0, le=1.0)

    accepted: bool


class RAGSource(BaseModel):
    """One document source used by RAG."""

    doc_id: str

    chunk_id: str

    similarity: float = Field(ge=-1.0, le=1.0)


class AssistantResponse(BaseModel):
    """Final structured response returned by the assistant."""

    query_type: Literal['intent', 'document']

    route: Literal['distilbert_high_confidence', 'low_confidence_consensus', 'document_rag', 'human_review']

    answer: str = Field(min_length=1)

    needs_human_review: bool

    intent: Optional[IntentResult] = None

    sources: List[RAGSource] = Field(default_factory=list)


def validate_response(data: dict) -> AssistantResponse:
    """Validate and normalize assistant output."""

    return AssistantResponse.model_validate(data)