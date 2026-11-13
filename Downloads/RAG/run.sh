#!/usr/bin/env bash
# Convenience wrapper around RUN_COMMANDS.md — Git Bash / WSL / Linux / macOS.
set -euo pipefail

PY=".venv/Scripts/python.exe"
if [ ! -f "$PY" ]; then
  PY=".venv/bin/python"
fi
if [ ! -f "$PY" ]; then
  echo "No virtualenv found. Create it first (see RUN_COMMANDS.md section 0)." >&2
  exit 1
fi

case "${1:-all}" in
  setup)
    "$PY" -m pip install -r requirements.txt
    ;;
  ingest)
    "$PY" -m src.rag.ingestion
    ;;
  baseline)
    "$PY" -m src.evaluation.run_evaluation --config configs/experiments/exp_001_baseline.yaml
    ;;
  enhanced)
    "$PY" -m src.evaluation.run_evaluation --config configs/experiments/exp_002_enhanced.yaml
    ;;
  reports)
    "$PY" -m src.evaluation.text_report --experiment-id EXP-001
    ;;
  api)
    "$PY" -m src.api.main
    ;;
  test)
    "$PY" -m pytest tests -q
    ;;
  all)
    "$PY" -m src.evaluation.dataset data/datasets/benchmark_v1.jsonl
    "$PY" -m src.rag.ingestion
    "$PY" -m src.evaluation.run_evaluation --config configs/experiments/exp_001_baseline.yaml
    "$PY" -m src.evaluation.run_evaluation --config configs/experiments/exp_002_enhanced.yaml
    ;;
  *)
    echo "usage: ./run.sh [setup|ingest|baseline|enhanced|reports|api|test|all]" >&2
    exit 2
    ;;
esac
