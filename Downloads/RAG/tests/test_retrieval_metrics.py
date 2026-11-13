import pytest

from src.evaluation.retrieval_metrics import compute_retrieval_metrics


def test_perfect_retrieval():
    metrics = compute_retrieval_metrics(["pricing", "support_sla"], ["pricing"], k=5)
    assert metrics.evaluable
    assert metrics.hit is True
    assert metrics.hit_rate == 1.0
    assert metrics.recall_at_1 == 1.0
    assert metrics.mrr == 1.0
    assert metrics.context_precision == 1.0
    assert metrics.context_recall == 1.0


def test_miss():
    metrics = compute_retrieval_metrics(["pricing", "api_limits"], ["support_sla", "data_retention"], k=5)
    assert metrics.hit is False
    assert metrics.hit_rate == 0.0
    assert metrics.context_recall == 0.0
    assert metrics.mrr == 0.0


def test_partial_recall_and_rank():
    metrics = compute_retrieval_metrics(["api_limits", "pricing"], ["pricing", "refund_policy"], k=5)
    assert metrics.hit is True
    assert metrics.recall_at_1 == 0.0
    assert metrics.recall_at_3 == 0.5
    assert metrics.context_recall == 0.5
    assert metrics.mrr == pytest.approx(0.5)


def test_duplicate_documents_are_deduplicated():
    metrics = compute_retrieval_metrics(["pricing", "pricing", "pricing"], ["pricing"], k=5)
    assert metrics.precision_at_k == 1.0
    assert metrics.recall_at_1 == 1.0


def test_unanswerable_questions_are_not_evaluated():
    metrics = compute_retrieval_metrics(["pricing"], [], k=5)
    assert metrics.evaluable is False
    assert metrics.hit is False
