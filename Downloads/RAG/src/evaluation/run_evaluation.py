"""Run a full RAG evaluation experiment and persist the results.

CLI:
    python -m src.evaluation.run_evaluation --config configs/experiments/exp_001_baseline.yaml
    python -m src.evaluation.run_evaluation --config <cfg> --limit 3 --skip-judge --no-report
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.core.config import Settings, load_settings
from src.core.logging import get_logger, setup_logging
from src.core.schemas import (
    EvaluationDatasetItem,
    ExperimentSummary,
    GenerationMetrics,
    QuestionResult,
)
from src.db import repository
from src.db.session import init_engine, session_scope
from src.evaluation.dataset import dataset_statistics, load_dataset, validate_expected_context
from src.evaluation.deepeval_adapter import run_deepeval
from src.evaluation.error_analysis import classify_error
from src.evaluation.geval import GevalJudge
from src.evaluation.retrieval_metrics import compute_retrieval_metrics
from src.evaluation.scoring import aggregate, passes_thresholds, weighted_score
from src.evaluation.text_metrics import compute_generation_metrics
from src.rag.embeddings import get_embedding_model
from src.rag.ingestion import load_documents
from src.rag.pipeline import RAGPipeline

log = get_logger(__name__)


def evaluate_item(
    item: EvaluationDatasetItem,
    pipeline: RAGPipeline,
    judge: GevalJudge,
    settings: Settings,
    embed,
    skip_judge: bool = False,
) -> QuestionResult:
    response = pipeline.answer(item.question, item.id)
    retrieved_doc_ids = [r.chunk.doc_id for r in response.retrieved]

    retrieval = compute_retrieval_metrics(retrieved_doc_ids, item.expected_context, k=settings.retrieval.top_k)

    if response.error:
        log.warning("{} failed: {}", item.id, response.error)
        generation = GenerationMetrics()
        geval = None
        deepeval_results = []
    else:
        generation = compute_generation_metrics(
            question=item.question,
            answer=response.answer,
            contexts=response.contexts,
            ground_truth=item.ground_truth,
            expected_context=item.expected_context,
            embed=embed,
            evaluation_cfg=settings.evaluation,
        )
        geval = None if skip_judge else judge.evaluate(
            item.question, response.answer, item.ground_truth, response.contexts
        )
        if geval is not None and not geval.judge_model:
            geval.judge_model = settings.evaluation.judge.model
        deepeval_results = run_deepeval(
            cfg=settings.evaluation.deepeval,
            judge_cfg=settings.evaluation.judge,
            question=item.question,
            answer=response.answer,
            contexts=response.contexts,
            ground_truth=item.ground_truth,
            expected_context=item.expected_context,
        )

    score, components = weighted_score(generation, retrieval, settings.evaluation.weights)
    passed = passes_thresholds(score, generation, retrieval, settings.evaluation.thresholds)
    analysis = classify_error(
        error=response.error,
        expected_context=item.expected_context,
        retrieval=retrieval,
        generation=generation,
        overall_score=score,
        thresholds=settings.evaluation.thresholds,
        is_refusal=generation.is_refusal,
    )
    # A refusal on an unanswerable question is a success, not a failure.
    if not item.expected_context and generation.is_refusal and not response.error:
        analysis = analysis.model_copy(update={"category": "none", "failed": False, "explanation": ""})

    return QuestionResult(
        question_id=item.id,
        question=item.question,
        ground_truth=item.ground_truth,
        category=item.category,
        difficulty=item.difficulty,
        expected_context=item.expected_context,
        answer=response.answer,
        retrieval=retrieval,
        generation=generation,
        deepeval=deepeval_results,
        geval=geval,
        error_analysis=analysis,
        overall_score=score,
        overall_components=components,
        passed=bool(passed),
        latency_ms=response.latency_ms,
        retrieval_latency_ms=response.retrieval_latency_ms,
        rerank_latency_ms=response.rerank_latency_ms,
        generation_latency_ms=response.generation_latency_ms,
        retrieved_ids=[r.chunk.chunk_id for r in response.retrieved],
        retrieved_sources=response.sources,
        contexts=response.contexts,
        prompt_version=response.prompt_version,
        llm=response.llm,
        error=response.error,
    )


def run_experiment(
    config_path: str | Path | None = None,
    dataset_path: str | Path | None = None,
    limit: int | None = None,
    skip_judge: bool = False,
    make_report: bool = True,
) -> tuple[ExperimentSummary, list[QuestionResult]]:
    settings = load_settings(config_path)
    setup_logging(settings.paths_resolved.logs, settings.logging.level)

    dataset_file = Path(dataset_path) if dataset_path else settings.paths.resolve(settings.experiment.dataset_path)
    items = load_dataset(dataset_file)
    if limit:
        items = items[:limit]

    documents = load_documents(settings.paths_resolved.documents)
    warnings = validate_expected_context(items, {str(doc.metadata["doc_id"]) for doc in documents})
    for warning in warnings:
        log.warning(warning)

    stats = dataset_statistics(items)
    log.info(
        "starting {} on {} questions {}",
        settings.experiment.experiment_id,
        len(items),
        stats,
    )

    init_engine(settings.paths_resolved.database)
    started_at = datetime.now(timezone.utc)

    with session_scope() as session:
        dataset_row = repository.get_or_create_dataset(
            session,
            name=dataset_file.stem,
            version=settings.experiment.dataset_version,
            path=str(dataset_file),
            num_questions=len(items),
        )
        experiment_row = repository.upsert_experiment(
            session,
            experiment_id=settings.experiment.experiment_id,
            name=settings.experiment.name,
            description=settings.experiment.description,
            config=settings.experiment.to_payload(settings),
            dataset_id=dataset_row.id,
            status="running",
        )
        experiment_pk = experiment_row.id

    pipeline = RAGPipeline(settings)
    pipeline.retriever.ensure_index()
    log.info("retriever ready: {}", pipeline.retriever.describe())
    embed = get_embedding_model(settings.embedding)
    judge = GevalJudge(settings.evaluation.judge)

    results: list[QuestionResult] = []
    for index, item in enumerate(items, start=1):
        result = evaluate_item(item, pipeline, judge, settings, embed, skip_judge=skip_judge)
        results.append(result)
        log.info(
            "[{}/{}] {} score={} cat={} hit={} latency={:.0f}ms",
            index,
            len(items),
            item.id,
            "n/a" if result.overall_score is None else f"{result.overall_score:.2f}",
            result.error_analysis.category,
            result.retrieval.hit,
            result.latency_ms,
        )

    metrics = aggregate(results)
    finished_at = datetime.now(timezone.utc)

    notes: list[str] = []
    if skip_judge:
        notes.append("G-Eval judge was skipped for this run (--skip-judge).")
    elif judge.enabled:
        notes.append(
            f"G-Eval is LLM-as-a-judge using {settings.evaluation.judge.model}; "
            f"scores available for {metrics.geval_coverage:.0%} of questions."
        )
    else:
        notes.append("G-Eval judge disabled in configuration; no LLM-judged scores in this run.")
    if not settings.evaluation.deepeval.enabled:
        notes.append(
            "DeepEval metrics were disabled for this run (evaluation.deepeval.enabled=false); "
            "no DeepEval scores are reported."
        )
    if warnings:
        notes.extend(warnings)

    report_dir = settings.paths_resolved.reports / "experiments"
    report_dir.mkdir(parents=True, exist_ok=True)
    results_path = report_dir / f"{settings.experiment.experiment_id}_results.json"
    results_path.write_text(
        json.dumps([json.loads(r.model_dump_json()) for r in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = ExperimentSummary(
        experiment_id=settings.experiment.experiment_id,
        name=settings.experiment.name,
        description=settings.experiment.description,
        dataset_id=dataset_file.stem,
        dataset_version=settings.experiment.dataset_version,
        num_questions=len(results),
        config=settings.experiment.to_payload(settings),
        metrics=metrics,
        started_at=started_at,
        finished_at=finished_at,
        notes=notes,
    )

    report_path: Path | None = None
    from src.evaluation.text_report import write_metrics_report, write_report

    metrics_path = write_metrics_report(
        summary, report_dir / f"{settings.experiment.experiment_id}_metrics.txt"
    )
    log.info("metrics written to {}", metrics_path)

    if make_report:
        report_path = report_dir / f"{settings.experiment.experiment_id}_report.txt"
        write_report(summary, results, report_path)
        summary.report_path = str(report_path)

    summary_path = report_dir / f"{settings.experiment.experiment_id}_summary.json"
    summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    with session_scope() as session:
        repository.save_results(session, experiment_pk, results)
        repository.finish_experiment(session, experiment_pk, summary)
        repository.register_report(
            session, experiment_pk, str(report_path or results_path), report_type="txt" if report_path else "json"
        )

    log.info(
        "{} complete: overall={:.3f} pass_rate={:.1%} hallucination_rate={:.1%}",
        summary.experiment_id,
        metrics.overall_score,
        metrics.pass_rate,
        metrics.hallucination_rate,
    )
    if report_path:
        log.info("report written to {}", report_path)
    return summary, results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a RAG-Bench evaluation experiment")
    parser.add_argument("--config", required=True, help="experiment config (merged over configs/config.yaml)")
    parser.add_argument("--dataset", default=None, help="override dataset path")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions")
    parser.add_argument("--skip-judge", action="store_true", help="skip the G-Eval LLM judge")
    parser.add_argument("--no-report", action="store_true", help="do not generate the text report")
    args = parser.parse_args()

    summary, _ = run_experiment(
        config_path=args.config,
        dataset_path=args.dataset,
        limit=args.limit,
        skip_judge=args.skip_judge,
        make_report=not args.no_report,
    )
    print(
        json.dumps(
            {
                "experiment_id": summary.experiment_id,
                "questions": summary.num_questions,
                "overall_score": round(summary.metrics.overall_score, 4),
                "pass_rate": round(summary.metrics.pass_rate, 4),
                "hit_rate": round(summary.metrics.hit_rate, 4),
                "context_recall": round(summary.metrics.context_recall, 4),
                "faithfulness": round(summary.metrics.faithfulness, 4),
                "hallucination_rate": round(summary.metrics.hallucination_rate, 4),
                "report": summary.report_path,
                "metrics": (
                    str(Path(summary.report_path).parent / f"{summary.experiment_id}_metrics.txt")
                    if summary.report_path
                    else None
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
