import pytest
from pydantic import ValidationError

from src.core.config import MetricWeights, load_settings


def test_default_weights_sum_to_one():
    settings = load_settings()
    total = sum(settings.evaluation.weights.model_dump().values())
    assert abs(total - 1.0) < 1e-9


def test_weights_must_sum_to_one():
    with pytest.raises(ValidationError):
        MetricWeights(faithfulness=0.5, answer_relevancy=0.5, context_recall=0.5,
                      context_precision=0.1, correctness=0.1)


def test_hybrid_config_turns_on_bm25():
    settings = load_settings("configs/experiments/exp_002_enhanced.yaml")
    assert settings.retrieval.strategy == "hybrid"
    assert settings.bm25.enabled is True
    assert settings.hybrid.enabled is True
    assert settings.reranker.enabled is True
    assert settings.generation.prompt_version == "v2"


def test_baseline_disables_enhancements():
    settings = load_settings("configs/experiments/exp_001_baseline.yaml")
    assert settings.retrieval.strategy == "vector"
    assert settings.bm25.enabled is False
    assert settings.reranker.enabled is False
    assert settings.generation.prompt_version == "v1"


def test_candidate_k_must_cover_top_k():
    with pytest.raises(ValidationError):
        from src.core.config import RetrievalConfig

        RetrievalConfig(top_k=10, candidate_k=5)
