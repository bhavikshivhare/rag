from pathlib import Path

import pytest

from src.evaluation.dataset import DatasetError, dataset_statistics, load_dataset, validate_expected_context

DATASET = Path(__file__).resolve().parents[1] / "data" / "datasets" / "benchmark_v1.jsonl"
DOCS = Path(__file__).resolve().parents[1] / "data" / "documents"


def test_dataset_loads_and_has_expected_shape():
    items = load_dataset(DATASET)
    assert len(items) == 23
    stats = dataset_statistics(items)
    assert stats["by_category"]["unanswerable"] == 2
    assert {"factual", "comparison", "multi-hop", "summarization"}.issubset(stats["by_category"])
    assert all(item.ground_truth for item in items)


def test_expected_context_points_at_real_documents():
    from src.rag.ingestion import load_documents

    items = load_dataset(DATASET)
    doc_ids = {doc.metadata["doc_id"] for doc in load_documents(DOCS)}
    assert validate_expected_context(items, doc_ids) == []


def test_duplicate_ids_are_rejected(tmp_path):
    path = tmp_path / "dup.jsonl"
    line = '{"id": "Q1", "question": "q", "ground_truth": "a"}'
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(DatasetError):
        load_dataset(path)


def test_missing_fields_are_rejected(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "Q1", "question": "q"}\n', encoding="utf-8")
    with pytest.raises(DatasetError):
        load_dataset(path)
