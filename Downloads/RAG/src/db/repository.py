"""Repository helpers used by the CLI, the report generator and the API."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.core.schemas import ExperimentSummary, QuestionResult
from src.db.models import DatasetRow, EvaluationResultRow, ExperimentRow, ReportRow


def _now() -> datetime:
    return datetime.now(timezone.utc)


def get_or_create_dataset(
    session: Session, *, name: str, version: str, path: str, num_questions: int
) -> DatasetRow:
    existing = session.scalar(
        select(DatasetRow).where(DatasetRow.name == name, DatasetRow.version == version)
    )
    if existing:
        existing.num_questions = num_questions
        existing.path = path
        return existing
    row = DatasetRow(name=name, version=version, path=path, num_questions=num_questions)
    session.add(row)
    session.flush()
    return row


def upsert_experiment(
    session: Session,
    *,
    experiment_id: str,
    name: str,
    description: str,
    config: dict[str, Any],
    dataset_id: int | None,
    status: str = "running",
) -> ExperimentRow:
    row = session.scalar(select(ExperimentRow).where(ExperimentRow.experiment_id == experiment_id))
    if row is None:
        row = ExperimentRow(
            experiment_id=experiment_id,
            name=name,
            description=description,
            config_json=config,
            dataset_id=dataset_id,
            status=status,
            started_at=_now(),
        )
        session.add(row)
    else:
        row.name = name
        row.description = description
        row.config_json = config
        row.dataset_id = dataset_id
        row.status = status
        row.started_at = _now()
        row.finished_at = None
        session.query(EvaluationResultRow).filter(EvaluationResultRow.experiment_id == row.id).delete()
    session.flush()
    return row


def save_results(session: Session, experiment_pk: int, results: list[QuestionResult]) -> None:
    rows = [
        EvaluationResultRow(
            experiment_id=experiment_pk,
            question_id=result.question_id,
            question=result.question,
            ground_truth=result.ground_truth,
            answer=result.answer,
            category=result.category,
            difficulty=result.difficulty,
            overall_score=result.overall_score,
            passed=1 if result.passed else 0,
            error_category=result.error_analysis.category,
            latency_ms=result.latency_ms,
            metrics_json=json.loads(result.model_dump_json()),
        )
        for result in results
    ]
    session.add_all(rows)


def finish_experiment(session: Session, experiment_pk: int, summary: ExperimentSummary) -> None:
    row = session.get(ExperimentRow, experiment_pk)
    if row is None:
        raise ValueError(f"experiment {experiment_pk} not found")
    row.metrics_json = summary.metrics.model_dump()
    row.summary_json = json.loads(summary.model_dump_json())
    row.status = summary.status
    row.finished_at = summary.finished_at
    row.report_path = summary.report_path


def register_report(session: Session, experiment_pk: int, path: str, report_type: str = "experiment") -> None:
    session.add(ReportRow(experiment_id=experiment_pk, path=path, report_type=report_type))


def get_experiment(session: Session, experiment_id: str) -> ExperimentRow | None:
    return session.scalar(select(ExperimentRow).where(ExperimentRow.experiment_id == experiment_id))


def list_experiments(session: Session) -> list[ExperimentRow]:
    return list(session.scalars(select(ExperimentRow).order_by(ExperimentRow.started_at.desc())))


def get_results(session: Session, experiment_id: str) -> list[EvaluationResultRow]:
    experiment = get_experiment(session, experiment_id)
    if experiment is None:
        return []
    return list(
        session.scalars(
            select(EvaluationResultRow)
            .where(EvaluationResultRow.experiment_id == experiment.id)
            .order_by(EvaluationResultRow.question_id)
        )
    )


def get_summary(session: Session, experiment_id: str) -> ExperimentSummary | None:
    experiment = get_experiment(session, experiment_id)
    if experiment is None:
        return None
    if experiment.summary_json:
        return ExperimentSummary.model_validate(experiment.summary_json)
    return None


def load_results_as_models(session: Session, experiment_id: str) -> list[QuestionResult]:
    return [QuestionResult.model_validate(row.metrics_json) for row in get_results(session, experiment_id)]
