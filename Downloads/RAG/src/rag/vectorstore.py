"""Vector store: LangChain FAISS with max-inner-product scoring.

Vectors are L2-normalised by the embedding model, so inner product equals cosine
similarity and a higher score is a better match. The index is persisted with
``save_local``/``load_local``; ``chunks.json`` keeps the project's chunk objects next to
it so a cached index can be inspected without loading FAISS.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.embeddings import Embeddings

from src.core.schemas import DocumentChunk
from src.rag.documents import to_chunk, to_document


def _as_list(vector) -> list[float]:
    return vector.tolist() if hasattr(vector, "tolist") else [float(value) for value in vector]


class FAISSVectorStore:
    """Thin adapter so callers keep working with :class:`DocumentChunk` objects."""

    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self._store: FAISS | None = None
        self.chunks: list[DocumentChunk] = []

    @property
    def store(self) -> FAISS:
        if self._store is None:
            raise RuntimeError("vector index is empty - build or load it first")
        return self._store

    def build(self, chunks: list[DocumentChunk], vectors=None) -> None:
        if not chunks:
            raise ValueError("cannot build a vector index over zero chunks")
        if vectors is None:
            vectors = self.embeddings.embed_documents([chunk.content for chunk in chunks])

        pairs = [(chunk.content, _as_list(vector)) for chunk, vector in zip(chunks, vectors)]
        self._store = FAISS.from_embeddings(
            text_embeddings=pairs,
            embedding=self.embeddings,
            metadatas=[to_document(chunk).metadata for chunk in chunks],
            ids=[chunk.chunk_id for chunk in chunks],
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )
        self.chunks = list(chunks)

    def search(self, query_vector, top_k: int) -> list[tuple[DocumentChunk, float]]:
        results = self.store.similarity_search_with_score_by_vector(
            _as_list(query_vector), k=min(top_k, len(self.chunks))
        )
        return [(to_chunk(document), float(score)) for document, score in results]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.store.save_local(str(directory))
        (directory / "chunks.json").write_text(
            json.dumps([chunk.model_dump() for chunk in self.chunks], ensure_ascii=False),
            encoding="utf-8",
        )

    def load(self, directory: Path) -> "FAISSVectorStore":
        index_path = directory / "index.faiss"
        chunks_path = directory / "chunks.json"
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f"incomplete index cache in {directory}")
        self._store = FAISS.load_local(
            str(directory),
            self.embeddings,
            allow_dangerous_deserialization=True,  # our own cache directory, written by save()
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )
        self.chunks = [
            DocumentChunk.model_validate(item) for item in json.loads(chunks_path.read_text(encoding="utf-8"))
        ]
        return self
