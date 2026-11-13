"""Plain-text evaluation report and the persistent baseline store.

The project writes no PDF, DOCX, Excel or HTML report files. One evaluation
writes a readable ``.txt`` summary, a numbers-only ``*_metrics.txt`` file and their
JSON artefacts; the saved baseline is only ever replaced when it is explicitly
requested.

CLI:
    python -m src.evaluation.text_report --experiment-id EXP-001
    python -m src.evaluation.text_report --experiment-id EXP-001 --save-baseline
    python -m src.evaluation.text_report --show-baseline
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.core.config import Settings, load_settings
from src.core.logging import get_logger, setup_logging
from src.core.schemas import ExperimentSummary, QuestionResult
from src.evaluation.scoring import aggregate_by

log = get_logger(__name__)

BASELINE_TXT = "baseline.txt"
BASELINE_JSON = "baseline.json"

FAILURE_LABELS = {
    "retrieval_failure": "retrieval failure",
    "generation_failure": "generation failure",
    "hallucination": "hallucination",
    "insufficient_context": "insufficient context",
    "evaluation_failure": "evaluation failure",
    "other": "other",
}

RULE = "-" * 78
TOP = "=" * 78


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #
def num(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def pct(value: float | None, digits: int = 1) -> str:
    return "N/A" if value is None else f"{value * 100:.{digits}f}%"


def seconds(milliseconds: float | None) -> str:
    return "N/A" if milliseconds is None else f"{milliseconds / 1000:.2f} s"


def _field(label: str, value: str, width: int = 21) -> str:
    return f"{label:<{width}}: {value}"


# --------------------------------------------------------------------------- #
# report sections
# --------------------------------------------------------------------------- #
def _configuration_lines(summary: ExperimentSummary) -> list[str]:
    cfg = summary.config or {}
    strategy = str(cfg.get("retrieval_strategy") or "unknown")
    fusion = cfg.get("hybrid_fusion")
    if strategy == "hybrid":
        strategy_text = f"hybrid (BM25 + vector, {fusion or 'rrf'} fusion)"
    elif strategy == "bm25":
        strategy_text = "bm25 (lexical only)"
    else:
        strategy_text = f"vector (FAISS cosine, top_k={cfg.get('top_k', 'N/A')})"

    reranker = cfg.get("reranker_model")
    if reranker:
        reranker_text = f"{reranker} (top {cfg.get('candidate_k', 'N/A')} -> {cfg.get('reranker_top_n', 'N/A')})"
    else:
        reranker_text = "N/A (disabled)"

    return [
        _field("Retrieval Strategy", strategy_text),
        _field("Embedding Model", str(cfg.get("embedding_model") or "unknown")),
        _field(
            "LLM",
            f"{cfg.get('llm', 'unknown')} (temperature {cfg.get('temperature', 'N/A')})",
        ),
        _field("Reranker", reranker_text),
        _field("Prompt Version", str(cfg.get("prompt_version") or "N/A")),
        _field(
            "Chunking",
            f"{cfg.get('chunk_size', 'N/A')} chars, {cfg.get('chunk_overlap', 'N/A')} overlap",
        ),
        _field("Candidates / Top K", f"{cfg.get('candidate_k', 'N/A')} / {cfg.get('top_k', 'N/A')}"),
    ]


def _result_lines(summary: ExperimentSummary) -> list[str]:
    m = summary.metrics
    passed = round(m.pass_rate * m.total_questions)
    return [
        _field("Faithfulness", num(m.faithfulness)),
        _field("Answer Relevancy", num(m.answer_relevancy)),
        _field("Contextual Precision", num(m.context_precision)),
        _field("Contextual Recall", num(m.context_recall)),
        _field("Answer Correctness", num(m.correctness)),
        _field("Overall Score", f"{num(m.overall_score)} (weighted, 0-1)"),
        _field("Threshold Pass Rate", f"{pct(m.pass_rate)} ({passed}/{m.total_questions})"),
        _field("Hallucination Rate", pct(m.hallucination_rate)),
        _field("Questions Answered", f"{m.answered_questions}/{m.total_questions}"),
        _field("Mean Latency", f"{seconds(m.avg_latency_ms)} (p95 {seconds(m.p95_latency_ms)})"),
    ]


def build_observations(summary: ExperimentSummary, results: list[QuestionResult]) -> list[str]:
    """Facts derived from measured values only — no targets, no invented gains."""
    m = summary.metrics
    observations: list[str] = []
    passed = round(m.pass_rate * m.total_questions)

    observations.append(
        f"Overall weighted score is {num(m.overall_score)} on a 0-1 scale, and "
        f"{passed} of {m.total_questions} questions pass every configured threshold ({pct(m.pass_rate)})."
    )
    observations.append(
        f"Retrieval returned at least one expected document for {pct(m.hit_rate)} of answerable "
        f"questions; Recall@1 is {pct(m.recall_at_1)} and MRR is {num(m.mrr)}."
    )
    observations.append(
        f"Faithfulness (deterministic sentence-support proxy) is {pct(m.faithfulness)} and "
        f"hallucination rate is {pct(m.hallucination_rate)} "
        f"({m.failure_counts.get('hallucination', 0)} question(s) classified as hallucination)."
    )
    observations.append(
        f"Answer correctness averages {num(m.correctness)} across answered questions; "
        f"token F1 against the reference answers is {num(m.token_f1)}."
    )

    by_category = aggregate_by(results, "category") if results else {}
    if by_category:
        weakest = min(by_category.items(), key=lambda kv: kv[1].overall_score)
        observations.append(
            f"Weakest question category: {weakest[0]} with an overall score of "
            f"{num(weakest[1].overall_score)} across {weakest[1].total_questions} question(s)."
        )

    failing = [r for r in results if not r.passed]
    if failing:
        listed = ", ".join(
            f"{r.question_id} ({FAILURE_LABELS.get(r.error_analysis.category, r.error_analysis.category)})"
            if r.error_analysis.category != "none"
            else f"{r.question_id} (below threshold, no error class)"
            for r in failing[:5]
        )
        more = "" if len(failing) <= 5 else f" and {len(failing) - 5} more"
        observations.append(f"Questions below threshold: {listed}{more}.")
    else:
        observations.append("No question fell below the configured thresholds in this run.")

    if m.geval_coverage and m.geval_coverage < 1.0:
        observations.append(
            f"The LLM judge produced usable scores for {pct(m.geval_coverage)} of questions; "
            "judge gaps are reported rather than filled with defaults."
        )
    return observations


def build_recommendations(summary: ExperimentSummary, results: list[QuestionResult]) -> list[str]:
    """Each suggestion is tied to a measured signal; none is emitted when the signal is fine."""
    m = summary.metrics
    recommendations: list[str] = []

    if m.hallucination_rate > 0:
        ids = ", ".join(
            r.question_id for r in results
            if r.error_analysis.category == "hallucination"
        ) or "the affected question(s)"
        recommendations.append(
            f"Add an explicit grounding and refusal rule to the prompt: {ids} were answered although "
            "the corpus does not support them."
        )
    if m.context_recall < 0.99:
        recommendations.append(
            f"Improve context coverage (currently {pct(m.context_recall)}): raise top_k or enable hybrid "
            "BM25 retrieval for questions that need more than one document."
        )
    if m.recall_at_1 < 0.90:
        recommendations.append(
            f"Reorder retrieved chunks (Recall@1 is only {pct(m.recall_at_1)}): a cross-encoder reranker "
            "over the candidate pool moves the best chunk to rank 1."
        )
    if m.correctness < 0.80:
        weakest = sorted(
            (r for r in results if r.overall_score is not None),
            key=lambda r: r.generation.correctness if r.generation.correctness is not None else 1.0,
        )[:3]
        ids = ", ".join(r.question_id for r in weakest) or "the lowest-scoring questions"
        recommendations.append(
            f"Tighten answer content (correctness {num(m.correctness)}): {ids} score lowest — instruct the "
            "model to state the exact requested fact instead of surrounding context."
        )
    if m.avg_latency_ms and m.p95_latency_ms > 20_000:
        recommendations.append(
            f"Reduce latency (p95 {seconds(m.p95_latency_ms)}): lower max_new_tokens or run generation on GPU; "
            "CPU inference dominates the per-question cost."
        )
    if m.faithfulness < 0.85:
        recommendations.append(
            f"Raise grounding (faithfulness {pct(m.faithfulness)}): keep the retrieved-context-only "
            "instruction and shorten answers so each claim stays traceable to a chunk."
        )

    recommendations.append(
        f"Grow the benchmark beyond {m.total_questions} questions before treating any change as real: on a "
        "set this size a single question moves aggregate metrics by several points."
    )
    return recommendations


def build_text_report(
    summary: ExperimentSummary,
    results: list[QuestionResult] | None = None,
) -> str:
    """Render the whole report as plain text (this is the only report format)."""
    results = results or []
    m = summary.metrics
    finished = summary.finished_at.astimezone(timezone.utc)

    lines: list[str] = [
        TOP,
        "RAG SYSTEM EVALUATION SUMMARY",
        TOP,
        "",
        _field("Evaluation Date", finished.strftime("%Y-%m-%d %H:%M UTC")),
        _field("Experiment", f"{summary.experiment_id} - {summary.name}"),
        _field("Dataset", f"{summary.dataset_id} (version {summary.dataset_version})"),
        _field("Total Questions", str(m.total_questions)),
        "",
        RULE,
        "CONFIGURATION",
        RULE,
        *_configuration_lines(summary),
        "",
        RULE,
        "EVALUATION RESULTS",
        RULE,
        *_result_lines(summary),
        "",
        RULE,
        "OBSERVATIONS",
        RULE,
    ]
    lines += [f"- {item}" for item in build_observations(summary, results)]

    lines += ["", RULE, "RECOMMENDATIONS", RULE]
    lines += [f"- {item}" for item in build_recommendations(summary, results)]

    if summary.notes:
        lines += ["", RULE, "RUN NOTES", RULE]
        lines += [f"- {note}" for note in summary.notes]

    lines += [
        "",
        RULE,
        f"Generated by RAG-Bench on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.",
        "All values above were measured by this run; missing values are shown as N/A.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# files
# --------------------------------------------------------------------------- #
def write_report(
    summary: ExperimentSummary,
    results: list[QuestionResult] | None,
    output_path: str | Path,
) -> Path:
    """Write the text report for one run. Never touches the baseline."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_text_report(summary, results), encoding="utf-8")
    return path


def _section(title: str) -> list[str]:
    return [RULE, title, RULE]


def build_metrics_report(summary: ExperimentSummary) -> str:
    """Render only the measured numbers - no observations, no recommendations."""
    m = summary.metrics
    finished = summary.finished_at.astimezone(timezone.utc)

    lines: list[str] = [
        TOP,
        "RAG EVALUATION METRICS",
        TOP,
        "",
        _field("Evaluation Date", finished.strftime("%Y-%m-%d %H:%M UTC")),
        _field("Experiment", f"{summary.experiment_id} - {summary.name}"),
        _field("Dataset", f"{summary.dataset_id} (version {summary.dataset_version})"),
        _field("Total Questions", str(m.total_questions)),
        "",
        *_section("RETRIEVAL"),
        _field("Hit Rate", pct(m.hit_rate)),
        _field("Recall@1", pct(m.recall_at_1)),
        _field("Recall@3", pct(m.recall_at_3)),
        _field("Recall@5", pct(m.recall_at_5)),
        _field("Recall@10", pct(m.recall_at_10)),
        _field("Precision@K", num(m.precision_at_k)),
        _field("MRR", num(m.mrr)),
        _field("Contextual Precision", num(m.context_precision)),
        _field("Contextual Recall", num(m.context_recall)),
        "",
        *_section("GENERATION"),
        _field("Faithfulness", num(m.faithfulness)),
        _field("Answer Relevancy", num(m.answer_relevancy)),
        _field("Answer Correctness", num(m.correctness)),
        _field("Token F1", num(m.token_f1)),
    ]

    judge = [
        ("G-Eval Correctness", m.geval_correctness),
        ("G-Eval Completeness", m.geval_completeness),
        ("G-Eval Groundedness", m.geval_groundedness),
        ("G-Eval Clarity", m.geval_clarity),
        ("G-Eval Conciseness", m.geval_conciseness),
    ]
    judge = [(label, value) for label, value in judge if value is not None]
    if judge or m.geval_coverage:
        lines += ["", *_section("LLM JUDGE")]
        lines += [_field(label, num(value)) for label, value in judge]
        lines.append(_field("Judge Coverage", pct(m.geval_coverage)))

    deepeval = [
        ("DeepEval Faithfulness", m.deepeval_faithfulness),
        ("DeepEval Answer Relevancy", m.deepeval_answer_relevancy),
        ("DeepEval Contextual Relevancy", m.deepeval_contextual_relevancy),
        ("DeepEval Contextual Recall", m.deepeval_contextual_recall),
    ]
    deepeval = [(label, value) for label, value in deepeval if value is not None]
    if deepeval:
        lines += ["", *_section("DEEPEVAL")]
        lines += [_field(label, num(value)) for label, value in deepeval]

    lines += [
        "",
        *_section("OVERALL"),
        _field("Overall Score", f"{num(m.overall_score)} (weighted, 0-1)"),
        _field(
            "Threshold Pass Rate",
            f"{pct(m.pass_rate)} ({round(m.pass_rate * m.total_questions)}/{m.total_questions})",
        ),
        _field("Hallucination Rate", pct(m.hallucination_rate)),
        _field("Questions Answered", f"{m.answered_questions}/{m.total_questions}"),
        "",
        *_section("LATENCY"),
        _field("Mean", seconds(m.avg_latency_ms)),
        _field("p50", seconds(m.p50_latency_ms)),
        _field("p95", seconds(m.p95_latency_ms)),
        _field("p99", seconds(m.p99_latency_ms)),
        _field("Retrieval (mean)", seconds(m.avg_retrieval_latency_ms)),
        _field("Rerank (mean)", seconds(m.avg_rerank_latency_ms)),
        _field("Generation (mean)", seconds(m.avg_generation_latency_ms)),
        "",
        *_section("FAILURE CLASSIFICATION"),
    ]
    failures = {key: value for key, value in (m.failure_counts or {}).items() if value}
    if failures:
        lines += [
            f"- {'no error class' if key == 'none' else FAILURE_LABELS.get(key, key)}: {value}"
            for key, value in sorted(failures.items())
        ]
    else:
        lines += ["- no error classes assigned"]
    lines += [
        "",
        RULE,
        "Numbers only: N/A means the metric was not computed for this run.",
        "",
    ]
    return "\n".join(lines)


def write_metrics_report(summary: ExperimentSummary, output_path: str | Path) -> Path:
    """Write the metrics-only file for one run (alongside the full report)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_metrics_report(summary), encoding="utf-8")
    return path


def baseline_paths(settings: Settings | None = None) -> tuple[Path, Path]:
    settings = settings or load_settings()
    directory = settings.paths_resolved.reports / "baseline"
    return directory / BASELINE_TXT, directory / BASELINE_JSON


def save_baseline(
    summary: ExperimentSummary,
    results: list[QuestionResult] | None = None,
    settings: Settings | None = None,
) -> Path:
    """Persist this run as the baseline. Only called on an explicit request."""
    settings = settings or load_settings()
    txt_path, json_path = baseline_paths(settings)
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    text = build_text_report(summary, results)
    txt_path.write_text(text, encoding="utf-8")
    payload = {
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment_id": summary.experiment_id,
        "name": summary.name,
        "overall_score": summary.metrics.overall_score,
        "num_questions": summary.num_questions,
        "summary": json.loads(summary.model_dump_json()),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("baseline saved from {} to {}", summary.experiment_id, txt_path)
    return txt_path


def load_baseline(settings: Settings | None = None) -> dict | None:
    """Return the saved baseline ({saved_at, experiment_id, summary, text}) or None."""
    txt_path, json_path = baseline_paths(settings)
    if not txt_path.exists() or not json_path.exists():
        return None
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.error("baseline metadata is not valid JSON: {}", json_path)
        return None
    payload["text"] = txt_path.read_text(encoding="utf-8")
    return payload


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Write the plain-text evaluation report from stored results")
    parser.add_argument("--experiment-id", default=None, help="stored experiment to render, e.g. EXP-001")
    parser.add_argument("--config", default=None, help="optional experiment config for settings")
    parser.add_argument("--output", default=None, help="output .txt path (default reports/experiments/<id>_report.txt)")
    parser.add_argument("--metrics", action="store_true", help="write only the metrics file (<id>_metrics.txt)")
    parser.add_argument("--save-baseline", action="store_true", help="also store this run as the baseline")
    parser.add_argument("--show-baseline", action="store_true", help="print the saved baseline and exit")
    args = parser.parse_args()

    settings = load_settings(args.config)
    setup_logging(settings.paths_resolved.logs, settings.logging.level)

    if args.show_baseline:
        baseline = load_baseline(settings)
        if baseline is None:
            raise SystemExit("no baseline saved yet; run with --experiment-id <ID> --save-baseline")
        print(baseline["text"])
        return

    if not args.experiment_id:
        parser.error("--experiment-id is required unless --show-baseline is used")

    from src.db import repository
    from src.db.session import init_engine, session_scope

    init_engine(settings.paths_resolved.database)
    with session_scope() as session:
        summary = repository.get_summary(session, args.experiment_id)
        if summary is None:
            raise SystemExit(f"experiment {args.experiment_id} not found in the database")
        results = repository.load_results_as_models(session, args.experiment_id)

    if args.metrics:
        metrics_path = write_metrics_report(
            summary, settings.paths_resolved.reports / "experiments" / f"{args.experiment_id}_metrics.txt"
        )
        print(f"metrics: {metrics_path}")
        print(build_metrics_report(summary))
        return

    output = Path(args.output) if args.output else (
        settings.paths_resolved.reports / "experiments" / f"{args.experiment_id}_report.txt"
    )
    path = write_report(summary, results, output)
    summary.report_path = str(path)

    with session_scope() as session:
        row = repository.get_experiment(session, args.experiment_id)
        if row is not None:
            repository.register_report(session, row.id, str(path), report_type="txt")

    if args.save_baseline:
        save_baseline(summary, results, settings)

    print(f"report: {path}")
    print(build_text_report(summary, results))


if __name__ == "__main__":
    main()
