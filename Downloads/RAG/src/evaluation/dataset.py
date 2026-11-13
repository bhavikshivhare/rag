"""Benchmark dataset loading and validation (JSONL)."""

from __future__ import annotations

import json
from pathlib import Path

from src.core.schemas import EvaluationDatasetItem

REQUIRED_FIELDS = {"id", "question", "ground_truth"}
KNOWN_CATEGORIES = {"factual", "comparison", "multi-hop", "summarization", "unanswerable"}
KNOWN_DIFFICULTIES = {"easy", "medium", "hard"}


class DatasetError(ValueError):
    pass


def load_dataset(path: str | Path) -> list[EvaluationDatasetItem]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise DatasetError(f"dataset not found: {dataset_path}")

    items: list[EvaluationDatasetItem] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{dataset_path.name}:{line_number} is not valid JSON: {exc}") from exc

        missing = REQUIRED_FIELDS - payload.keys()
        if missing:
            raise DatasetError(f"{dataset_path.name}:{line_number} missing fields: {sorted(missing)}")
        if payload["id"] in seen:
            raise DatasetError(f"duplicate question id {payload['id']} at line {line_number}")
        seen.add(payload["id"])

        payload.setdefault("expected_context", [])
        payload.setdefault("category", "factual")
        payload.setdefault("difficulty", "medium")
        if payload["category"] not in KNOWN_CATEGORIES:
            raise DatasetError(f"unknown category '{payload['category']}' for {payload['id']}")
        if payload["difficulty"] not in KNOWN_DIFFICULTIES:
            raise DatasetError(f"unknown difficulty '{payload['difficulty']}' for {payload['id']}")

        items.append(EvaluationDatasetItem.model_validate(payload))

    if not items:
        raise DatasetError(f"dataset {dataset_path} is empty")
    return items


def dataset_statistics(items: list[EvaluationDatasetItem]) -> dict[str, dict[str, int]]:
    by_category: dict[str, int] = {}
    by_difficulty: dict[str, int] = {}
    for item in items:
        by_category[item.category] = by_category.get(item.category, 0) + 1
        by_difficulty[item.difficulty] = by_difficulty.get(item.difficulty, 0) + 1
    return {"by_category": by_category, "by_difficulty": by_difficulty}


def validate_expected_context(items: list[EvaluationDatasetItem], known_doc_ids: set[str]) -> list[str]:
    """Return warnings for expected_context ids that do not exist in the corpus."""
    warnings: list[str] = []
    for item in items:
        unknown = [doc for doc in item.expected_context if doc not in known_doc_ids]
        if unknown:
            warnings.append(f"{item.id}: expected_context references unknown doc ids {unknown}")
    return warnings


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validate a benchmark dataset")
    parser.add_argument("path", nargs="?", default="data/datasets/benchmark_v1.jsonl")
    args = parser.parse_args()
    items = load_dataset(args.path)
    stats = dataset_statistics(items)
    print(json.dumps({"questions": len(items), **stats}, indent=2))


if __name__ == "__main__":
    main()
