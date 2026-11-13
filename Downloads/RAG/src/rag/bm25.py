"""Lexical retrieval through LangChain's ``BM25Retriever``.

The retriever hands back ranked documents; the underlying ``rank_bm25`` vectorizer is
kept reachable so the raw scores can be reported alongside the vector scores (they are
what ``hybrid.fusion: weighted`` normalises).
"""

from __future__ import annotations

import re

from langchain_community.retrievers import BM25Retriever

from src.core.schemas import DocumentChunk
from src.rag.documents import to_chunk, to_document

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, chunks: list[DocumentChunk], k1: float = 1.5, b: float = 0.75):
        if not chunks:
            raise ValueError("cannot build a BM25 index over zero chunks")
        self.chunks = list(chunks)
        self._retriever = BM25Retriever.from_documents(
            [to_document(chunk) for chunk in self.chunks],
            bm25_params={"k1": k1, "b": b},
            preprocess_func=tokenize,
        )

    @property
    def retriever(self) -> BM25Retriever:
        return self._retriever

    def search(self, query: str, top_k: int) -> list[tuple[DocumentChunk, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        # The vectorizer's corpus order is the constructor order, so scores line up with self.chunks.
        scores = self._retriever.vectorizer.get_scores(tokens)
        score_by_id = {chunk.chunk_id: float(scores[index]) for index, chunk in enumerate(self.chunks)}

        self._retriever.k = min(top_k, len(self.chunks))
        results: list[tuple[DocumentChunk, float]] = []
        for document in self._retriever.invoke(query):
            chunk = to_chunk(document)
            score = score_by_id.get(chunk.chunk_id, 0.0)
            if score > 0:
                results.append((chunk, score))
        return results[:top_k]
