"""Document RAG interfaces.

The evaluation pipeline used:
1. semantic retrieval over document chunks;
2. top-k grounded context;
3. LLM answer generation;
4. independent offline QA evaluation.

The API key and provider configuration are not stored here.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class DocumentChunk:
    doc_id: str
    chunk_id: str
    text: str
    similarity: float


@dataclass
class RAGResult:
    answer: str
    sources: List[DocumentChunk]


def build_context(chunks: List[DocumentChunk]) -> str:
    """Build grounded context from retrieved document chunks."""

    blocks = []

    for chunk in chunks:
        blocks.append(
            f"[DOC_ID: {chunk.doc_id}]\n"
            f"[CHUNK_ID: {chunk.chunk_id}]\n"
            f"{chunk.text}"
        )

    return "\n\n---\n\n".join(blocks)