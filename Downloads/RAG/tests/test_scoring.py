from src.core.config import MetricWeights, ThresholdsConfig
from src.core.schemas import GenerationMetrics, RetrievalMetrics
from src.evaluation.error_analysis import classify_error
from src.evaluation.scoring import passes_thresholds, weighted_score


def _generation(**overrides):
    base = dict(faithfulness=0.9, answer_relevancy=0.8, correctness=0.8, token_f1=0.6,
                unsupported_sentences=0, total_sentences=4, is_refusal=False)
    base.update(overrides)
    return GenerationMetrics(**base)


def _retrieval(**overrides):
    base = dict(evaluable=True, hit=True, hit_rate=1.0, recall_at_1=1.0, recall_at_3=1.0,
                recall_at_5=1.0, recall_at_10=1.0, precision_at_k=0.5, mrr=1.0,
                context_precision=0.8, context_recall=1.0, relevant_retrieved=2,
                expected_retrieved=2, expected_total=2)
    base.update(overrides)
    return RetrievalMetrics(**base)


def test_weighted_score_uses_configured_weights():
    score, components = weighted_score(_generation(), _retrieval(), MetricWeights())
    expected = (0.9 * 0.30 + 0.8 * 0.20 + 1.0 * 0.20 + 0.8 * 0.15 + 0.8 * 0.15)
    assert abs(score - expected) < 1e-6
    assert set(components) == {"faithfulness", "answer_relevancy", "context_recall",
                              "context_precision", "correctness"}


def test_missing_components_are_renormalised_not_defaulted():
    score, components = weighted_score(
        _generation(faithfulness=None, answer_relevancy=None),
        _retrieval(evaluable=False),
        MetricWeights(),
    )
    assert "faithfulness" not in components
    assert score is not None
    assert abs(score - 0.8) < 1e-6


def test_thresholds_pass_and_fail():
    thresholds = ThresholdsConfig()
    assert passes_thresholds(0.9, _generation(), _retrieval(), thresholds) is True
    assert passes_thresholds(0.9, _generation(faithfulness=0.2), _retrieval(), thresholds) is False


def test_unanswerable_refusal_is_not_a_hallucination():
    generation = _generation(faithfulness=1.0, correctness=1.0, answer_relevancy=None,
                             is_refusal=True, total_sentences=1)
    analysis = classify_error(
        error=None, expected_context=[], retrieval=_retrieval(evaluable=False),
        generation=generation, overall_score=1.0, thresholds=ThresholdsConfig(), is_refusal=True,
    )
    assert analysis.category != "hallucination"


def test_unanswerable_non_refusal_is_hallucination():
    generation = _generation(faithfulness=None, correctness=0.0, answer_relevancy=None, is_refusal=False)
    analysis = classify_error(
        error=None, expected_context=[], retrieval=_retrieval(evaluable=False),
        generation=generation, overall_score=0.2, thresholds=ThresholdsConfig(), is_refusal=False,
    )
    assert analysis.category == "hallucination"
    assert analysis.failed is True


def test_retrieval_miss_is_classified_first():
    analysis = classify_error(
        error=None, expected_context=["pricing"], retrieval=_retrieval(context_recall=0.0, mrr=0.0, hit=False),
        generation=_generation(correctness=0.1), overall_score=0.2,
        thresholds=ThresholdsConfig(), is_refusal=False,
    )
    assert analysis.category == "retrieval_failure"
