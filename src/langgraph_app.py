"""End-to-end LangGraph orchestration for the assistant.

The graph coordinates:
- query type detection
- intent classification
- semantic/document retrieval
- optional additional search
- answer validation
- human-review fallback
- final response

ML models and external services are injected as adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph


# ------------------------------------------------------------
# State
# ------------------------------------------------------------

QueryType = Literal['intent', 'document']

class AssistantState(TypedDict, total=False):
    query: str
    query_type: QueryType

    intent_id: Optional[int]
    intent_name: Optional[str]
    classifier_confidence: Optional[float]

    retrieval_label: Optional[int]
    retrieval_similarity: Optional[float]

    retrieved_chunks: List[Dict[str, Any]]
    additional_chunks: List[Dict[str, Any]]

    generated_answer: Optional[str]

    answer_is_valid: bool
    needs_additional_search: bool
    needs_human_review: bool

    route: Optional[str]
    final_answer: Optional[str]


# ------------------------------------------------------------
# External ML/service adapters
# ------------------------------------------------------------

@dataclass
class AssistantAdapters:
    """Functions supplied by the application layer."""

    determine_query_type: Callable[[str], QueryType]

    classify_intent: Callable[[str], Dict[str, Any]]

    semantic_retrieve: Callable[[str], Dict[str, Any]]

    document_retrieve: Callable[[str, int], List[Dict[str, Any]]]

    generate_answer: Callable[[str, List[Dict[str, Any]]], str]

    validate_answer: Callable[[str, str, List[Dict[str, Any]]], bool]


# ------------------------------------------------------------
# Frozen experimental policy
# ------------------------------------------------------------

DISTILBERT_THRESHOLD = 0.65
RETRIEVAL_SIMILARITY_GATE = 1.0
RAG_TOP_K = 3
RAG_ADDITIONAL_TOP_K = 5


# ------------------------------------------------------------
# Graph builder
# ------------------------------------------------------------

def build_assistant_graph(adapters: AssistantAdapters):
    """Build and compile the assistant LangGraph."""

    def determine_query_type_node(
        state: AssistantState,
    ) -> AssistantState:

        query_type = adapters.determine_query_type(
            state['query']
        )

        return {
            'query_type': query_type,
        }

    def classify_intent_node(
        state: AssistantState,
    ) -> AssistantState:

        result = adapters.classify_intent(
            state['query']
        )

        return {
            'intent_id': result.get('intent_id'),
            'intent_name': result.get('intent_name'),
            'classifier_confidence': float(
                result.get('confidence', 0.0)
            ),
        }

    def semantic_retrieval_node(
        state: AssistantState,
    ) -> AssistantState:

        result = adapters.semantic_retrieve(
            state['query']
        )

        return {
            'retrieval_label': result.get('label'),
            'retrieval_similarity': float(
                result.get('similarity', 0.0)
            ),
        }

    def check_intent_node(
        state: AssistantState,
    ) -> AssistantState:

        confidence = float(
            state.get('classifier_confidence') or 0.0
        )

        classifier_label = state.get('intent_id')
        retrieval_label = state.get('retrieval_label')

        retrieval_similarity = float(
            state.get('retrieval_similarity') or 0.0
        )

        if confidence >= DISTILBERT_THRESHOLD:
            return {
                'needs_human_review': False,
                'route': 'distilbert_high_confidence',
            }

        safe_consensus = (
            classifier_label is not None
            and retrieval_label is not None
            and classifier_label == retrieval_label
            and retrieval_similarity
            >= RETRIEVAL_SIMILARITY_GATE
        )

        if safe_consensus:
            return {
                'needs_human_review': False,
                'route': 'low_confidence_consensus',
            }

        return {
            'needs_human_review': True,
            'route': 'human_review',
        }

    def form_intent_answer_node(
        state: AssistantState,
    ) -> AssistantState:

        intent_name = state.get('intent_name')

        return {
            'final_answer': (
                f'Intent: {intent_name}'
                if intent_name
                else 'Intent identified.'
            )
        }

    def document_retrieval_node(
        state: AssistantState,
    ) -> AssistantState:

        chunks = adapters.document_retrieve(
            state['query'],
            RAG_TOP_K,
        )

        return {
            'retrieved_chunks': chunks,
        }

    def generate_answer_node(
        state: AssistantState,
    ) -> AssistantState:

        chunks = state.get(
            'retrieved_chunks',
            [],
        )

        answer = adapters.generate_answer(
            state['query'],
            chunks,
        )

        return {
            'generated_answer': answer,
        }

    def validate_answer_node(
        state: AssistantState,
    ) -> AssistantState:

        chunks = state.get(
            'retrieved_chunks',
            [],
        )

        answer = state.get(
            'generated_answer'
        ) or ''

        valid = adapters.validate_answer(
            state['query'],
            answer,
            chunks,
        )

        return {
            'answer_is_valid': bool(valid),
            'needs_additional_search': not bool(valid),
        }

    def additional_search_node(
        state: AssistantState,
    ) -> AssistantState:

        chunks = adapters.document_retrieve(
            state['query'],
            RAG_ADDITIONAL_TOP_K,
        )

        return {
            'retrieved_chunks': chunks,
            'additional_chunks': chunks,
            'needs_additional_search': False,
        }

    def validate_after_additional_search_node(
        state: AssistantState,
    ) -> AssistantState:

        chunks = state.get(
            'retrieved_chunks',
            [],
        )

        answer = adapters.generate_answer(
            state['query'],
            chunks,
        )

        valid = adapters.validate_answer(
            state['query'],
            answer,
            chunks,
        )

        return {
            'generated_answer': answer,
            'answer_is_valid': bool(valid),
            'needs_human_review': not bool(valid),
            'route': (
                'document_rag'
                if valid
                else 'human_review'
            ),
        }

    def form_document_answer_node(
        state: AssistantState,
    ) -> AssistantState:

        return {
            'final_answer': state.get(
                'generated_answer'
            )
        }

    def human_review_node(
        state: AssistantState,
    ) -> AssistantState:

        return {
            'needs_human_review': True,
            'route': 'human_review',
            'final_answer': (
                'Request requires human review.'
            ),
        }


    def route_by_query_type(
        state: AssistantState,
    ) -> str:

        if state.get('query_type') == 'document':
            return 'document'
        return 'intent'


    def route_after_intent_check(
        state: AssistantState,
    ) -> str:

        if state.get('needs_human_review'):
            return 'human_review'
        return 'automatic'


    def route_after_rag_validation(
        state: AssistantState,
    ) -> str:

        if state.get('answer_is_valid'):
            return 'valid'
        return 'additional_search'


    def route_after_second_validation(
        state: AssistantState,
    ) -> str:

        if state.get('answer_is_valid'):
            return 'valid'
        return 'human_review'


    graph = StateGraph(AssistantState)

    graph.add_node(
        'determine_query_type',
        determine_query_type_node,
    )

    graph.add_node(
        'classify_intent',
        classify_intent_node,
    )

    graph.add_node(
        'semantic_retrieval',
        semantic_retrieval_node,
    )

    graph.add_node(
        'check_intent',
        check_intent_node,
    )

    graph.add_node(
        'form_intent_answer',
        form_intent_answer_node,
    )

    graph.add_node(
        'document_retrieval',
        document_retrieval_node,
    )

    graph.add_node(
        'generate_answer',
        generate_answer_node,
    )

    graph.add_node(
        'validate_answer',
        validate_answer_node,
    )

    graph.add_node(
        'additional_search',
        additional_search_node,
    )

    graph.add_node(
        'validate_after_additional_search',
        validate_after_additional_search_node,
    )

    graph.add_node(
        'form_document_answer',
        form_document_answer_node,
    )

    graph.add_node(
        'human_review',
        human_review_node,
    )


    graph.add_edge(
        START,
        'determine_query_type',
    )

    graph.add_conditional_edges(
        'determine_query_type',
        route_by_query_type,
        {
            'intent': 'classify_intent',
            'document': 'document_retrieval',
        },
    )

    graph.add_edge(
        'classify_intent',
        'semantic_retrieval',
    )

    graph.add_edge(
        'semantic_retrieval',
        'check_intent',
    )

    graph.add_conditional_edges(
        'check_intent',
        route_after_intent_check,
        {
            'automatic': 'form_intent_answer',
            'human_review': 'human_review',
        },
    )

    graph.add_edge(
        'form_intent_answer',
        END,
    )

    graph.add_edge(
        'document_retrieval',
        'generate_answer',
    )

    graph.add_edge(
        'generate_answer',
        'validate_answer',
    )

    graph.add_conditional_edges(
        'validate_answer',
        route_after_rag_validation,
        {
            'valid': 'form_document_answer',
            'additional_search': 'additional_search',
        },
    )

    graph.add_edge(
        'additional_search',
        'validate_after_additional_search',
    )

    graph.add_conditional_edges(
        'validate_after_additional_search',
        route_after_second_validation,
        {
            'valid': 'form_document_answer',
            'human_review': 'human_review',
        },
    )

    graph.add_edge(
        'form_document_answer',
        END,
    )

    graph.add_edge(
        'human_review',
        END,
    )

    return graph.compile()


# ------------------------------------------------------------
# Public helper
# ------------------------------------------------------------

def run_assistant(
    graph,
    query: str,
) -> AssistantState:
    """Run one query through the compiled graph."""

    return graph.invoke(
        {
            'query': query,
            'needs_human_review': False,
            'needs_additional_search': False,
            'answer_is_valid': False,
            'retrieved_chunks': [],
            'additional_chunks': [],
        }
    )