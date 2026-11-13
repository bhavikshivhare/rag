"""Hybrid retrieval.

Reciprocal rank fusion is delegated to LangChain's ``EnsembleRetriever`` (weights and
the RRF constant ``c`` come from config). LangChain has no min-max weighted ensemble, so
``weighted_score_fusion`` stays here as a small pure function used when
``hybrid.fusion: weighted`` is selected.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import EnsembleRetriever

from src.core.config import HybridConfig


def make_hybrid_retriever(
    vector_retriever: BaseRetriever,
    bm25_retriever: BaseRetriever,
    cfg: HybridConfig,
) -> BaseRetriever:
    """RRF over the vector and BM25 retrievers."""
    if cfg.fusion != "rrf":
        raise ValueError(f"fusion '{cfg.fusion}' is not a LangChain ensemble; use weighted_score_fusion()")
    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[cfg.vector_weight, cfg.bm25_weight],
        c=cfg.rrf_k,
    )


def _min_max(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if abs(high - low) < 1e-12:
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def weighted_score_fusion(
    vector_results: list[tuple[Document, float]],
    bm25_results: list[tuple[Document, float]],
    weights: dict[str, float],
) -> list[tuple[Document, float]]:
    """Min-max normalise each retriever's scores, then take the weighted sum."""
    rankings = {"vector": vector_results, "bm25": bm25_results}
    normalised: dict[str, dict[str, float]] = {}
    lookup: dict[str, Document] = {}
    for name, results in rankings.items():
        normalised[name] = _min_max({document.metadata["chunk_id"]: score for document, score in results})
        for document, _score in results:
            lookup.setdefault(document.metadata["chunk_id"], document)

    combined: dict[str, float] = {}
    for name, scores in normalised.items():
        weight = weights.get(name, 1.0)
        for chunk_id, score in scores.items():
            combined[chunk_id] = combined.get(chunk_id, 0.0) + weight * score

    ordered = sorted(combined.items(), key=lambda item: item[1], reverse=True)
    return [(lookup[chunk_id], score) for chunk_id, score in ordered]
