"""Intent classification interface.

The final experimental model is DistilBERT trained for
Banking77 intent classification.

Heavy model weights are intentionally not stored in GitHub.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ClassificationResult:
    intent_id: int
    intent_name: str
    confidence: float
    accepted: bool


def apply_reject_policy(
    intent_id: int,
    intent_name: str,
    confidence: float,
    threshold: float,
) -> ClassificationResult:
    """Apply the validation-selected reject threshold."""

    return ClassificationResult(
        intent_id=int(intent_id),
        intent_name=str(intent_name),
        confidence=float(confidence),
        accepted=bool(confidence >= threshold),
    )