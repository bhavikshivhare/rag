"""Deterministic retrieval metrics computed against doc-level relevance labels.

Relevance is judged at document granularity: a retrieved chunk is relevant when its
``doc_id`` appears in the item's ``expected_context``. This is deliberate - it does
not reward a chunk merely for containing the same words as the ground truth, and it
does not require an LLM to label relevance.

Questions with an empty ``expected_context`` (unanswerable questions) are marked
``evaluable=False`` and excluded from retrieval aggregates.
"""

from __future__ import annotations

from src.core.schemas import RetrievalMetrics


def _dedup_doc_ids(doc_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for doc_id in doc_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            ordered.append(doc_id)
    return ordered


def compute_retrieval_metrics(
    retrieved_doc_ids: list[str],
    expected_doc_ids: list[str],
    k: int = 5,
) -> RetrievalMetrics:
    retrieved = _dedup_doc_ids(retrieved_doc_ids)[:k]
    expected = set(expected_doc_ids)

    if not expected:
        return RetrievalMetrics(evaluable=False, relevant_retrieved=0, expected_retrieved=0, expected_total=0)

    relevance = [1 if doc_id in expected else 0 for doc_id in retrieved]
    relevant_retrieved = sum(relevance)
    hit = relevant_retrieved > 0

    first_relevant_rank = next((i + 1 for i, rel in enumerate(relevance) if rel), None)
    mrr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0

    def recall_at(n: int) -> float:
        if not expected:
            return 0.0
        return len({doc_id for doc_id in retrieved[:n] if doc_id in expected}) / len(expected)

    # RAGAS-style rank-weighted context precision.
    cumulative_relevant = 0
    precision_sum = 0.0
    for position, rel in enumerate(relevance, start=1):
        if rel:
            cumulative_relevant += 1
            precision_sum += cumulative_relevant / position
    denominator = min(len(retrieved), len(expected)) or 1
    context_precision = precision_sum / denominator if relevant_retrieved else 0.0

    return RetrievalMetrics(
        evaluable=True,
        hit=hit,
        hit_rate=1.0 if hit else 0.0,
        recall_at_1=recall_at(1),
        recall_at_3=recall_at(3),
        recall_at_5=recall_at(5),
        recall_at_10=recall_at(10),
        precision_at_k=relevant_retrieved / len(retrieved) if retrieved else 0.0,
        mrr=mrr,
        context_precision=min(context_precision, 1.0),
        context_recall=relevant_retrieved / len(expected),
        relevant_retrieved=relevant_retrieved,
        expected_retrieved=relevant_retrieved,
        expected_total=len(expected),
    )
