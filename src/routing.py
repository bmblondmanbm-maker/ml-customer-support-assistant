"""Frozen routing policy corresponding to the final experiment."""

from dataclasses import dataclass

from .config import (
    DISTILBERT_CONFIDENCE_THRESHOLD,
    RETRIEVAL_SIMILARITY_THRESHOLD,
)


@dataclass
class RoutingDecision:
    route: str
    auto_closed: bool
    final_label: int | None


def route_request(
    classifier_label: int,
    classifier_confidence: float,
    retrieval_label: int | None = None,
    retrieval_similarity: float | None = None,
) -> RoutingDecision:
    """Route one request according to the frozen policy.

    Routes:
    - high-confidence DistilBERT -> automatic close
    - low-confidence consensus + similarity gate -> automatic close
    - otherwise -> human review
    """

    if classifier_confidence >= DISTILBERT_CONFIDENCE_THRESHOLD:
        return RoutingDecision(
            route="distilbert_high_confidence",
            auto_closed=True,
            final_label=int(classifier_label),
        )

    if (
        retrieval_label is not None
        and retrieval_similarity is not None
        and int(retrieval_label) == int(classifier_label)
        and float(retrieval_similarity) >= RETRIEVAL_SIMILARITY_THRESHOLD
    ):
        return RoutingDecision(
            route="low_confidence_consensus",
            auto_closed=True,
            final_label=int(classifier_label),
        )

    return RoutingDecision(
        route="human_review",
        auto_closed=False,
        final_label=None,
    )