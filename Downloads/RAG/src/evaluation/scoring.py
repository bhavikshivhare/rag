"""Weighted overall scoring and aggregation.

Weights come from configs/evaluation.yaml and are validated to sum to 1.
Components that are genuinely unavailable (e.g. context recall for an unanswerable
question) are skipped and the remaining weights are renormalised - they are never
replaced by a default value.
"""

from __future__ import annotations

import numpy as np

from src.core.config import MetricWeights, ThresholdsConfig
from src.core.schemas import (
    AggregateMetrics,
    GenerationMetrics,
    QuestionResult,
    RetrievalMetrics,
)

WEIGHT_KEYS = ("faithfulness", "answer_relevancy", "context_recall", "context_precision", "correctness")


def weighted_score(
    generation: GenerationMetrics,
    retrieval: RetrievalMetrics,
    weights: MetricWeights,
) -> tuple[float | None, dict[str, float]]:
    values = {
        "faithfulness": generation.faithfulness,
        "answer_relevancy": generation.answer_relevancy,
        "context_recall": retrieval.context_recall if retrieval.evaluable else None,
        "context_precision": retrieval.context_precision if retrieval.evaluable else None,
        "correctness": generation.correctness,
    }
    components = {key: value for key, value in values.items() if value is not None}
    weight_map = weights.model_dump()

    if not components:
        return None, {}

    available_weight = sum(weight_map[key] for key in components)
    if available_weight <= 0:
        return None, {}

    score = sum(components[key] * weight_map[key] for key in components) / available_weight
    return float(max(0.0, min(1.0, score))), components


def passes_thresholds(
    score: float | None,
    generation: GenerationMetrics,
    retrieval: RetrievalMetrics,
    thresholds: ThresholdsConfig,
) -> bool:
    if score is None or score < thresholds.overall_pass:
        return False
    checks = [
        (generation.faithfulness, thresholds.faithfulness),
        (generation.answer_relevancy, thresholds.answer_relevancy),
        (generation.correctness, thresholds.correctness),
    ]
    if retrieval.evaluable:
        checks.append((retrieval.context_recall, thresholds.context_recall))
    return all(value is None or value >= threshold for value, threshold in checks)


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return float(np.mean(present)) if present else None


def aggregate(results: list[QuestionResult]) -> AggregateMetrics:
    if not results:
        return AggregateMetrics(total_questions=0)

    evaluable = [r for r in results if r.retrieval.evaluable]
    latencies = [r.latency_ms for r in results]
    answered = [r for r in results if r.answer and not r.error]

    def avg(metric: str, source: list[QuestionResult]) -> float | None:
        return _mean([getattr(r, metric) for r in source])

    retrieval_avg = lambda field: _mean([getattr(r.retrieval, field) for r in evaluable])  # noqa: E731

    failures: dict[str, int] = {}
    for result in results:
        category = result.error_analysis.category
        failures[category] = failures.get(category, 0) + 1

    return AggregateMetrics(
        total_questions=len(results),
        answered_questions=len(answered),
        hit_rate=retrieval_avg("hit_rate") or 0.0,
        recall_at_1=retrieval_avg("recall_at_1") or 0.0,
        recall_at_3=retrieval_avg("recall_at_3") or 0.0,
        recall_at_5=retrieval_avg("recall_at_5") or 0.0,
        recall_at_10=retrieval_avg("recall_at_10") or 0.0,
        precision_at_k=retrieval_avg("precision_at_k") or 0.0,
        mrr=retrieval_avg("mrr") or 0.0,
        context_precision=retrieval_avg("context_precision") or 0.0,
        context_recall=retrieval_avg("context_recall") or 0.0,
        faithfulness=_mean([r.generation.faithfulness for r in answered]) or 0.0,
        answer_relevancy=_mean([r.generation.answer_relevancy for r in answered]) or 0.0,
        correctness=_mean([r.generation.correctness for r in answered]) or 0.0,
        token_f1=_mean([r.generation.token_f1 for r in answered]) or 0.0,
        geval_correctness=_mean([r.gev_val("correctness") for r in results]),
        geval_completeness=_mean([r.gev_val("completeness") for r in results]),
        geval_groundedness=_mean([r.gev_val("groundedness") for r in results]),
        geval_clarity=_mean([r.gev_val("clarity") for r in results]),
        geval_conciseness=_mean([r.gev_val("conciseness") for r in results]),
        geval_coverage=(
            sum(1 for r in results if r.gev_val("correctness") is not None) / len(results) if results else 0.0
        ),
        deepeval_faithfulness=_mean(
            [next((m.score for m in r.deepeval if m.metric == "faithfulness"), None) for r in results]
        ),
        deepeval_answer_relevancy=_mean(
            [next((m.score for m in r.deepeval if m.metric == "answer_relevancy"), None) for r in results]
        ),
        deepeval_contextual_relevancy=_mean(
            [next((m.score for m in r.deepeval if m.metric == "contextual_relevancy"), None) for r in results]
        ),
        deepeval_contextual_recall=_mean(
            [next((m.score for m in r.deepeval if m.metric == "contextual_recall"), None) for r in results]
        ),
        overall_score=_mean([r.overall_score for r in results]) or 0.0,
        pass_rate=sum(1 for r in results if r.passed) / len(results),
        hallucination_rate=failures.get("hallucination", 0) / len(results),
        avg_latency_ms=float(np.mean(latencies)),
        p50_latency_ms=float(np.percentile(latencies, 50)),
        p95_latency_ms=float(np.percentile(latencies, 95)),
        p99_latency_ms=float(np.percentile(latencies, 99)),
        avg_retrieval_latency_ms=float(np.mean([r.retrieval_latency_ms for r in results])),
        avg_rerank_latency_ms=float(np.mean([r.rerank_latency_ms for r in results])),
        avg_generation_latency_ms=float(np.mean([r.generation_latency_ms for r in results])),
        failure_counts=failures,
    )


def aggregate_by(results: list[QuestionResult], key: str) -> dict[str, AggregateMetrics]:
    groups: dict[str, list[QuestionResult]] = {}
    for result in results:
        value = str(getattr(result, key))
        groups.setdefault(value, []).append(result)
    return {name: aggregate(group) for name, group in sorted(groups.items())}
