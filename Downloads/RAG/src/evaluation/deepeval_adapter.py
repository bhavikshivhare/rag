"""Optional DeepEval integration.

DeepEval is only used when ``evaluation.deepeval.enabled`` is true. It needs an LLM
judge; ``judge_provider=local`` wraps the configured local HF model, ``openai``
uses an OpenAI-compatible API key from the environment.

If DeepEval is not installed, no judge is available, or a metric call fails, the
failure is recorded on the result. Nothing is replaced by a default score.
"""

from __future__ import annotations

import os
from typing import Any

from src.core.config import DeepEvalConfig, JudgeConfig
from src.core.logging import get_logger
from src.core.schemas import DeepEvalMetricResult

log = get_logger(__name__)

METRIC_NAMES = ["faithfulness", "answer_relevancy", "contextual_relevancy", "contextual_recall"]


def is_available() -> bool:
    try:
        import deepeval  # noqa: F401
    except Exception:
        return False
    return True


def _local_llm(judge_cfg: JudgeConfig):
    """Build a DeepEvalBaseLLM adapter around the project's local HF client."""
    from deepeval.models import DeepEvalBaseLLM

    from src.rag.llm import get_llm

    class LocalJudge(DeepEvalBaseLLM):
        def __init__(self) -> None:
            self._llm = get_llm(judge_cfg, role="deepeval")

        def load_model(self):
            return self._llm

        def generate(self, prompt: str, schema=None):
            result = self._llm.generate([{"role": "user", "content": prompt}])
            if schema is not None:
                return schema.model_validate_json(result.text)
            return result.text

        async def a_generate(self, prompt: str, schema=None):
            return self.generate(prompt, schema)

        def get_model_name(self) -> str:
            return judge_cfg.model

    return LocalJudge()


def _build_metrics(cfg: DeepEvalConfig, judge_cfg: JudgeConfig) -> dict[str, Any]:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
    )

    model: Any = _local_llm(judge_cfg) if cfg.judge_provider == "local" else None
    common = {"model": model, "async_mode": False, "verbose_mode": False, "include_reason": True}
    return {
        "faithfulness": FaithfulnessMetric(threshold=cfg.thresholds.get("faithfulness", 0.7), **common),
        "answer_relevancy": AnswerRelevancyMetric(threshold=cfg.thresholds.get("answer_relevancy", 0.7), **common),
        "contextual_relevancy": ContextualRelevancyMetric(
            threshold=cfg.thresholds.get("contextual_relevancy", 0.7), **common
        ),
        "contextual_recall": ContextualRecallMetric(threshold=cfg.thresholds.get("contextual_recall", 0.7), **common),
    }


def run_deepeval(
    *,
    cfg: DeepEvalConfig,
    judge_cfg: JudgeConfig,
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
    expected_context: list[str],
) -> list[DeepEvalMetricResult]:
    if not cfg.enabled:
        return []
    if not is_available():
        return [
            DeepEvalMetricResult(metric=name, error_message="deepeval is not installed")
            for name in METRIC_NAMES
        ]
    if cfg.judge_provider == "openai" and not os.environ.get(cfg.api_key_env):
        return [
            DeepEvalMetricResult(
                metric=name,
                error_message=f"{cfg.api_key_env} is not set, so the DeepEval judge cannot run",
            )
            for name in METRIC_NAMES
        ]

    try:
        from deepeval.test_case import LLMTestCase

        metrics = _build_metrics(cfg, judge_cfg)
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=ground_truth,
            retrieval_context=contexts or [""],
            context=contexts or [""],
        )
    except Exception as exc:  # configuration/import problem
        return [
            DeepEvalMetricResult(metric=name, error_message=f"deepeval setup failed: {exc}")
            for name in METRIC_NAMES
        ]

    results: list[DeepEvalMetricResult] = []
    for name, metric in metrics.items():
        if name == "contextual_recall" and not expected_context:
            results.append(
                DeepEvalMetricResult(metric=name, error_message="no expected context for an unanswerable question")
            )
            continue
        try:
            metric.measure(test_case)
            results.append(
                DeepEvalMetricResult(
                    metric=name,
                    score=float(metric.score) if metric.score is not None else None,
                    threshold=cfg.thresholds.get(name),
                    passed=bool(metric.is_successful()) if metric.score is not None else None,
                    reason=getattr(metric, "reason", None),
                )
            )
        except Exception as exc:  # judge failure -> recorded, never defaulted
            log.warning("deepeval metric {} failed: {}", name, exc)
            results.append(DeepEvalMetricResult(metric=name, error_message=f"metric failed: {exc}"))
    return results
