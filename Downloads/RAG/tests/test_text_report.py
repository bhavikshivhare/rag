from datetime import datetime, timezone

from src.core.config import load_settings
from src.core.schemas import (
    ErrorAnalysis,
    ExperimentSummary,
    GenerationMetrics,
    QuestionResult,
    RetrievalMetrics,
)
from src.evaluation.scoring import aggregate
from src.evaluation.text_report import (
    build_text_report,
    load_baseline,
    save_baseline,
    write_report,
)


def _result(qid: str, score: float, category: str, error_category: str) -> QuestionResult:
    return QuestionResult(
        question_id=qid,
        question=f"Question {qid}?",
        ground_truth="Reference answer.",
        category=category,
        difficulty="easy",
        expected_context=["pricing"] if category != "unanswerable" else [],
        answer="The Growth plan costs $199 per month.",
        retrieval=RetrievalMetrics(
            evaluable=category != "unanswerable",
            hit=score > 0.5,
            hit_rate=1.0 if score > 0.5 else 0.0,
            recall_at_1=0.5,
            recall_at_3=1.0,
            recall_at_5=1.0,
            recall_at_10=1.0,
            precision_at_k=0.4,
            mrr=1.0,
            context_precision=0.7,
            context_recall=0.9,
            relevant_retrieved=1,
            expected_retrieved=1,
            expected_total=1,
        ),
        generation=GenerationMetrics(
            faithfulness=0.8, answer_relevancy=0.9, correctness=0.75, token_f1=0.6,
            unsupported_sentences=1, total_sentences=5, is_refusal=False,
        ),
        error_analysis=ErrorAnalysis(
            category=error_category, explanation="synthetic", failed=error_category != "none"
        ),
        overall_score=score,
        overall_components={"faithfulness": 0.8, "correctness": 0.75},
        passed=score > 0.6,
        latency_ms=5000.0,
        retrieval_latency_ms=200.0,
        generation_latency_ms=4800.0,
        retrieved_sources=["pricing / Plans"],
        prompt_version="v1",
        llm="test-model",
    )


def _summary(experiment_id: str, name: str, results: list[QuestionResult]) -> ExperimentSummary:
    now = datetime.now(timezone.utc)
    return ExperimentSummary(
        experiment_id=experiment_id,
        name=name,
        description="synthetic experiment",
        dataset_id="benchmark_v1",
        dataset_version="v1",
        num_questions=len(results),
        config={
            "embedding_model": "test-embed",
            "retrieval_strategy": "vector",
            "top_k": 5,
            "candidate_k": 20,
            "reranker_enabled": False,
            "reranker_model": None,
            "llm": "test-model",
            "temperature": 0.0,
            "prompt_version": "v1",
            "chunk_size": 700,
            "chunk_overlap": 80,
        },
        metrics=aggregate(results),
        started_at=now,
        finished_at=now,
        notes=["synthetic test run"],
    )


def _settings(tmp_path):
    settings = load_settings()
    settings.paths.reports = str(tmp_path)
    return settings


def test_text_report_contains_required_sections(tmp_path):
    results = [
        _result("Q1", 0.9, "factual", "none"),
        _result("Q2", 0.3, "multi-hop", "retrieval_failure"),
    ]
    summary = _summary("EXP-001", "Synthetic baseline", results)
    text = build_text_report(summary, results)

    for label in (
        "RAG SYSTEM EVALUATION SUMMARY",
        "Evaluation Date",
        "Dataset",
        "Total Questions",
        "CONFIGURATION",
        "Retrieval Strategy",
        "Embedding Model",
        "LLM",
        "Reranker",
        "Faithfulness",
        "Answer Relevancy",
        "Contextual Precision",
        "Contextual Recall",
        "Answer Correctness",
        "Overall Score",
        "OBSERVATIONS",
        "RECOMMENDATIONS",
    ):
        assert label in text, label

    # measured values are printed, not defaults
    assert f"{summary.metrics.faithfulness:.3f}" in text
    assert f"{summary.metrics.overall_score:.3f}" in text
    assert "test-embed" in text and "test-model" in text
    assert "benchmark_v1" in text
    # the failing question is named
    assert "Q2" in text


def test_report_file_is_written_as_text(tmp_path):
    results = [_result("Q1", 0.9, "factual", "none")]
    summary = _summary("EXP-001", "Baseline", results)
    path = write_report(summary, results, tmp_path / "reports" / "EXP-001_report.txt")

    assert path.exists()
    assert path.suffix == ".txt"
    assert path.read_text(encoding="utf-8").startswith("=" * 10)
    assert not list(path.parent.glob("*.pdf"))


def test_baseline_is_only_replaced_on_request(tmp_path):
    settings = _settings(tmp_path)
    first = _summary("EXP-001", "Baseline", [_result("Q1", 0.9, "factual", "none")])
    save_baseline(first, [], settings)

    baseline = load_baseline(settings)
    assert baseline is not None
    assert baseline["experiment_id"] == "EXP-001"
    assert "RAG SYSTEM EVALUATION SUMMARY" in baseline["text"]

    # a new run writes its own report and must not touch the baseline
    second = _summary("EXP-002", "Enhanced", [_result("Q1", 0.7, "factual", "none")])
    write_report(second, [], tmp_path / "experiments" / "EXP-002_report.txt")
    assert load_baseline(settings)["experiment_id"] == "EXP-001"

    # only an explicit save replaces it
    save_baseline(second, [], settings)
    assert load_baseline(settings)["experiment_id"] == "EXP-002"


def test_metrics_file_holds_the_measured_numbers(tmp_path):
    from src.evaluation.text_report import build_metrics_report, write_metrics_report

    summary = _summary("EXP-002", "Enhanced", [_result("Q1", 0.8, "factual", "none")])
    text = build_metrics_report(summary)
    assert "RAG EVALUATION METRICS" in text
    assert "RETRIEVAL" in text and "GENERATION" in text and "OVERALL" in text
    assert "Overall Score" in text
    assert "Observation" not in text and "Recommendation" not in text

    path = write_metrics_report(summary, tmp_path / "EXP-002_metrics.txt")
    assert path.exists()
    assert path.read_text(encoding="utf-8") == text
