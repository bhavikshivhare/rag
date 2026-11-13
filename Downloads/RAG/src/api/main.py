"""FastAPI service in front of the evaluation engine.

This is a programmatic interface for running benchmarks and fetching the text report
and metrics file - not an interactive chat dashboard.

CLI:
    python -m src.api.main            # uvicorn on 127.0.0.1:8000
    uvicorn src.api.main:app --reload
"""

from __future__ import annotations

import html
import shutil
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.core.config import PROJECT_ROOT, load_settings
from src.core.logging import get_logger, setup_logging
from src.db import repository
from src.db.session import get_db, init_engine
from src.evaluation.dataset import DatasetError, load_dataset
from src.evaluation.text_report import load_baseline, save_baseline, write_metrics_report, write_report

log = get_logger(__name__)
settings = load_settings()
setup_logging(settings.paths_resolved.logs, settings.logging.level)
init_engine(settings.paths_resolved.database)

app = FastAPI(
    title="RAG-Bench API",
    description="Run RAG evaluations and retrieve text reports.",
    version="1.0.0",
)

# experiment_id -> {"status":..., "detail":...}
JOBS: dict[str, dict[str, str]] = {}


class EvaluationRequest(BaseModel):
    experiment_id: str = Field(..., pattern=r"^EXP-[A-Za-z0-9_-]+$")
    config_path: str = "configs/experiments/exp_001_baseline.yaml"
    dataset_path: str | None = None
    limit: int | None = Field(default=None, ge=1)
    skip_judge: bool = False
    generate_report: bool = True


class BaselineRequest(BaseModel):
    experiment_id: str


class DatasetResponse(BaseModel):
    id: int
    name: str
    version: str
    path: str
    num_questions: int


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "app": settings.app.name,
        "embedding_model": settings.embedding.model,
        "default_llm": settings.generation.model,
        "database": str(settings.paths_resolved.database),
    }


@app.get("/", response_class=HTMLResponse)
def index(db: Session = Depends(get_db)) -> str:
    """Minimal page showing the saved baseline report and links to stored run reports."""
    payload = load_baseline(settings)
    if payload is None:
        body = (
            "<p>No baseline report saved yet. Run an evaluation, then store it with "
            "<code>POST /api/baseline {\"experiment_id\": \"EXP-001\"}</code>.</p>"
        )
    else:
        body = (
            f"<p><b>Baseline:</b> {html.escape(str(payload['experiment_id']))} "
            f"&middot; saved {html.escape(str(payload['saved_at']))}</p>"
            f"<pre>{html.escape(payload['text'])}</pre>"
        )
    return (
        '<!doctype html><html><head><meta charset="utf-8"><title>RAG-Bench</title>'
        "<style>body{font-family:system-ui,sans-serif;margin:0;padding:2rem;max-width:62rem}"
        "pre{background:#f5f5f5;padding:1rem;overflow-x:auto;font-size:12px;line-height:1.45}"
        "a{color:#1f3864}</style></head><body><h1>RAG-Bench</h1>"
        f"{body}"
        f"{_run_links(db)}"
        '<p><a href="/docs">API docs</a> &middot; <a href="/api/baseline">/api/baseline</a> '
        '&middot; <a href="/api/experiments">/api/experiments</a></p></body></html>'
    )


def _run_links(db: Session) -> str:
    """Plain list of stored experiments; each report opens as text."""
    rows = repository.list_experiments(db)
    if not rows:
        return ""
    items = "".join(
        f'<li><a href="/api/reports/{html.escape(row.experiment_id)}">{html.escape(row.experiment_id)}</a>'
        f" - {html.escape(row.name)} ({html.escape(row.status)})"
        f"</li>"
        for row in rows
    )
    return f"<h2>Stored runs</h2><ul>{items}</ul>"


# ---------------------------------------------------------------------------
# baseline (explicitly stored, never overwritten by a run)
# ---------------------------------------------------------------------------
@app.get("/api/baseline")
def get_baseline() -> dict:
    payload = load_baseline(settings)
    if payload is None:
        raise HTTPException(status_code=404, detail="no baseline saved yet")
    return {
        "experiment_id": payload["experiment_id"],
        "saved_at": payload["saved_at"],
        "overall_score": payload.get("overall_score"),
        "report": payload["text"],
    }


@app.post("/api/baseline")
def update_baseline(request: BaselineRequest, db: Session = Depends(get_db)) -> dict:
    """Replace the stored baseline with an existing experiment. Explicit request only."""
    summary = repository.get_summary(db, request.experiment_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"{request.experiment_id} not found")
    results = repository.load_results_as_models(db, request.experiment_id)
    path = save_baseline(summary, results, settings)
    return {"experiment_id": request.experiment_id, "baseline": str(path)}


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
@app.post("/api/datasets", response_model=DatasetResponse, status_code=201)
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename or not file.filename.lower().endswith((".jsonl", ".json")):
        raise HTTPException(status_code=400, detail="dataset must be a .jsonl file")
    target_dir = settings.paths_resolved.datasets / "uploads"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / Path(file.filename).name
    with target.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    try:
        items = load_dataset(target)
    except DatasetError as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    row = repository.get_or_create_dataset(
        db, name=target.stem, version="custom", path=str(target), num_questions=len(items)
    )
    db.commit()
    return DatasetResponse(id=row.id, name=row.name, version=row.version, path=row.path, num_questions=row.num_questions)


@app.get("/api/datasets", response_model=list[DatasetResponse])
def list_datasets(db: Session = Depends(get_db)):
    from src.db.models import DatasetRow

    return [
        DatasetResponse(id=row.id, name=row.name, version=row.version, path=row.path, num_questions=row.num_questions)
        for row in db.query(DatasetRow).order_by(DatasetRow.created_at.desc()).all()
    ]


# ---------------------------------------------------------------------------
# experiments / evaluations
# ---------------------------------------------------------------------------
def _run_job(experiment_id: str, config_path: str, dataset_path: str | None, limit: int | None,
             skip_judge: bool, make_report: bool) -> None:
    from src.evaluation.run_evaluation import run_experiment

    JOBS[experiment_id] = {"status": "running", "detail": "evaluation in progress"}
    try:
        summary, _ = run_experiment(
            config_path=config_path,
            dataset_path=dataset_path,
            limit=limit,
            skip_judge=skip_judge,
            make_report=make_report,
        )
        JOBS[experiment_id] = {
            "status": "completed",
            "detail": f"overall={summary.metrics.overall_score:.3f} report={summary.report_path}",
        }
    except Exception as exc:  # surfaced through the API, not swallowed
        log.exception("experiment {} failed", experiment_id)
        JOBS[experiment_id] = {"status": "failed", "detail": str(exc)}


def _queue(request, background: BackgroundTasks, limit: int | None) -> dict:
    """Validate the config, mark the job queued and hand it to the background task."""
    config_file = PROJECT_ROOT / request.config_path
    if not config_file.exists():
        raise HTTPException(status_code=404, detail=f"config not found: {request.config_path}")
    JOBS[request.experiment_id] = {"status": "queued", "detail": request.config_path}
    background.add_task(
        _run_job,
        request.experiment_id,
        request.config_path,
        request.dataset_path,
        limit,
        request.skip_judge,
        request.generate_report,
    )
    return {"experiment_id": request.experiment_id, "status": "queued", "job": JOBS[request.experiment_id]}


@app.post("/api/experiments", status_code=202)
def create_experiment(request: EvaluationRequest, background: BackgroundTasks):
    """Run the full dataset (same as /api/evaluations without a limit)."""
    return _queue(request, background, None)


@app.post("/api/evaluations", status_code=202)
def create_evaluation(request: EvaluationRequest, background: BackgroundTasks):
    return _queue(request, background, request.limit)


@app.get("/api/experiments")
def list_experiments(db: Session = Depends(get_db)):
    rows = repository.list_experiments(db)
    return [
        {
            "experiment_id": row.experiment_id,
            "name": row.name,
            "status": row.status,
            "started_at": row.started_at.isoformat() if row.started_at else None,
            "finished_at": row.finished_at.isoformat() if row.finished_at else None,
            "overall_score": (row.metrics_json or {}).get("overall_score"),
            "report_path": row.report_path,
            "job": JOBS.get(row.experiment_id),
        }
        for row in rows
    ]


@app.get("/api/experiments/{experiment_id}")
def get_experiment(experiment_id: str, db: Session = Depends(get_db)):
    summary = repository.get_summary(db, experiment_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"{experiment_id} not found")
    return {"summary": summary.model_dump(), "job": JOBS.get(experiment_id)}


@app.get("/api/evaluations/{experiment_id}")
def get_evaluation(experiment_id: str, db: Session = Depends(get_db)):
    summary = repository.get_summary(db, experiment_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"{experiment_id} not found")
    return {"experiment_id": experiment_id, "status": summary.status, "metrics": summary.metrics.model_dump(),
            "job": JOBS.get(experiment_id)}


@app.get("/api/evaluations/{experiment_id}/results")
def get_evaluation_results(experiment_id: str, db: Session = Depends(get_db)):
    results = repository.get_results(db, experiment_id)
    if not results:
        raise HTTPException(status_code=404, detail=f"no results for {experiment_id}")
    return [
        {
            "question_id": row.question_id,
            "question": row.question,
            "answer": row.answer,
            "overall_score": row.overall_score,
            "passed": bool(row.passed),
            "error_category": row.error_category,
            "latency_ms": row.latency_ms,
            "metrics": row.metrics_json,
        }
        for row in results
    ]


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------
@app.get("/api/reports/{experiment_id}", response_class=PlainTextResponse)
def get_report(experiment_id: str, db: Session = Depends(get_db)):
    """The plain-text report for a stored experiment (regenerated from stored results if absent)."""
    summary = repository.get_summary(db, experiment_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"{experiment_id} not found")
    default_path = settings.paths_resolved.reports / "experiments" / f"{experiment_id}_report.txt"
    stored = summary.report_path or ""
    path = Path(stored) if stored.endswith(".txt") else default_path
    if not path.exists():
        results = repository.load_results_as_models(db, experiment_id)
        write_report(summary, results, path)
    return PlainTextResponse(path.read_text(encoding="utf-8"))


@app.get("/api/metrics/{experiment_id}", response_class=PlainTextResponse)
def get_metrics(experiment_id: str, db: Session = Depends(get_db)):
    """The metrics-only text file for a stored experiment (regenerated if absent)."""
    summary = repository.get_summary(db, experiment_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"{experiment_id} not found")
    path = settings.paths_resolved.reports / "experiments" / f"{experiment_id}_metrics.txt"
    if not path.exists():
        write_metrics_report(summary, path)
    return PlainTextResponse(path.read_text(encoding="utf-8"))


def main() -> None:
    import uvicorn

    uvicorn.run("src.api.main:app", host=settings.api.host, port=settings.api.port, reload=False)


if __name__ == "__main__":
    main()
