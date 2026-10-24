"""Semantic retrieval utilities.

The project uses normalized sentence embeddings and cosine
similarity over a retrieval corpus built only from train_part.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class RetrievalResult:
    label: int
    label_name: str
    similarity: float
    text: str


def consensus_is_safe(
    classifier_label: int,
    retrieval_label: int,
    retrieval_similarity: float,
    similarity_threshold: float,
) -> bool:
    """Check whether low-confidence consensus may auto-close.

    In the final experiment the selected similarity threshold
    was 1.0, so no additional low-confidence requests were
    safely auto-closed under the >=95% accuracy constraint.
    """

    return (
        int(classifier_label) == int(retrieval_label)
        and float(retrieval_similarity) >= float(similarity_threshold)
    )