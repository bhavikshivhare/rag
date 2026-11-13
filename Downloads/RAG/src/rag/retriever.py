"""Retrieval orchestration over LangChain components.

The vector retriever, the BM25 retriever, the ``EnsembleRetriever`` and the
``CrossEncoderReranker`` all come from LangChain; this module wires them from config and
converts the resulting documents back into the project's ``RetrievedChunk`` schema.
The index is cached on disk (keyed by embedding model + corpus + chunking config) so
repeated runs of different experiments share one embedding pass.
"""

from __future__ import annotations

import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.core.config import Settings
from src.core.logging import get_logger
from src.core.schemas import DocumentChunk, RetrievedChunk
from src.rag.bm25 import BM25Index
from src.rag.chunking import slugify
from src.rag.documents import to_chunk, to_document
from src.rag.embeddings import get_embedding_model
from src.rag.hybrid import make_hybrid_retriever, weighted_score_fusion
from src.rag.ingestion import corpus_fingerprint
from src.rag.reranker import get_compressor
from src.rag.vectorstore import FAISSVectorStore

log = get_logger(__name__)


class IndexCache:
    """Disk layout: data/cache/index/<model>__<corpus fingerprint>/"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = settings.paths_resolved.cache / "index"
        self.key = f"{slugify(settings.embedding.model)}__{corpus_fingerprint(settings.paths_resolved.documents, settings.chunking)}"
        self.path = self.root / self.key

    def exists(self) -> bool:
        return (self.path / "index.faiss").exists() and (self.path / "chunks.json").exists()

    def describe(self) -> dict[str, str | bool]:
        return {"cache_key": self.key, "path": str(self.path), "hit": self.exists()}


class RetrieverManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cache = IndexCache(settings)
        self.embeddings = get_embedding_model(settings.embedding)
        self.store = FAISSVectorStore(self.embeddings)
        self.chunks: list[DocumentChunk] = []
        self._bm25: BM25Index | None = None
        self._bm25_key: str | None = None
        self._hybrid: BaseRetriever | None = None
        self._hybrid_key: str | None = None
        self._ready = False
        self.last_timings: dict[str, float] = {"embedding": 0.0, "search": 0.0, "rerank": 0.0}

    # ------------------------------------------------------------------ index
    def ensure_index(self, chunks: list[DocumentChunk] | None = None) -> None:
        if self._ready:
            return
        if self.cache.exists():
            self.store.load(self.cache.path)
            self.chunks = self.store.chunks
            log.info("loaded vector index from cache: {}", self.cache.key)
        else:
            if chunks is None:
                from src.rag.ingestion import ingest

                chunks = ingest(self.settings)
            started = time.perf_counter()
            self.store.build(chunks)
            self.cache.path.mkdir(parents=True, exist_ok=True)
            self.store.save(self.cache.path)
            self.chunks = self.store.chunks
            log.info("built vector index with {} chunks in {:.1f}s", len(self.chunks), time.perf_counter() - started)
        self._ready = True

    def _bm25_index(self) -> BM25Index:
        key = f"{self.cache.key}:{self.settings.bm25.k1}:{self.settings.bm25.b}"
        if self._bm25 is None or self._bm25_key != key:
            self._bm25 = BM25Index(self.chunks, k1=self.settings.bm25.k1, b=self.settings.bm25.b)
            self._bm25_key = key
        return self._bm25

    def _hybrid_retriever(self) -> BaseRetriever:
        s = self.settings
        key = (
            f"{self.cache.key}|{s.hybrid.fusion}|{s.hybrid.rrf_k}|{s.hybrid.vector_weight}|"
            f"{s.hybrid.bm25_weight}|{s.bm25.k1}|{s.bm25.b}|{s.retrieval.candidate_k}"
        )
        if self._hybrid is None or self._hybrid_key != key:
            vector_retriever = self.store.store.as_retriever(search_kwargs={"k": s.retrieval.candidate_k})
            bm25_retriever = self._bm25_index().retriever
            bm25_retriever.k = s.retrieval.candidate_k
            self._hybrid = make_hybrid_retriever(vector_retriever, bm25_retriever, s.hybrid)
            self._hybrid_key = key
        return self._hybrid

    # -------------------------------------------------------------- retrieval
    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        strategy: str | None = None,
        use_reranker: bool | None = None,
    ) -> list[RetrievedChunk]:
        self.ensure_index()
        s = self.settings
        strategy = strategy or s.retrieval.strategy
        top_k = top_k or s.retrieval.top_k
        reranker_on = s.reranker.enabled if use_reranker is None else use_reranker
        candidate_k = max(s.retrieval.candidate_k, top_k)

        timings = {"embedding": 0.0, "search": 0.0, "rerank": 0.0}
        documents: list[Document] = []
        scores: dict[str, float] = {}
        vector_pairs: list[tuple[DocumentChunk, float]] = []
        search_started = time.perf_counter()

        weighted = strategy == "hybrid" and s.hybrid.fusion == "weighted"
        if strategy == "vector" or weighted:
            embed_started = time.perf_counter()
            query_vector = self.embeddings.embed_query(question)
            timings["embedding"] = time.perf_counter() - embed_started
            vector_pairs = self.store.search(query_vector, candidate_k)

        if strategy == "vector":
            documents = [to_document(chunk) for chunk, _score in vector_pairs]
            scores = {chunk.chunk_id: score for chunk, score in vector_pairs}
        elif strategy == "bm25":
            bm25_pairs = self._bm25_index().search(question, candidate_k)
            documents = [to_document(chunk) for chunk, _score in bm25_pairs]
            scores = {chunk.chunk_id: score for chunk, score in bm25_pairs}
        elif weighted:
            bm25_pairs = self._bm25_index().search(question, candidate_k)
            fused = weighted_score_fusion(
                [(to_document(chunk), score) for chunk, score in vector_pairs],
                [(to_document(chunk), score) for chunk, score in bm25_pairs],
                {"vector": s.hybrid.vector_weight, "bm25": s.hybrid.bm25_weight},
            )
            documents = [document for document, _score in fused][:candidate_k]
            scores = {document.metadata["chunk_id"]: score for document, score in fused}
        else:
            documents = self._hybrid_retriever().invoke(question)[:candidate_k]
            # EnsembleRetriever returns documents in fused order but not the fused value,
            # so report the RRF weight of the rank each document ended at.
            scores = {
                document.metadata["chunk_id"]: 1.0 / (s.hybrid.rrf_k + rank)
                for rank, document in enumerate(documents, start=1)
            }
        timings["search"] = time.perf_counter() - search_started

        if strategy == "vector" and s.retrieval.similarity_threshold > 0:
            documents = [
                document
                for document in documents
                if scores.get(document.metadata["chunk_id"], 0.0) >= s.retrieval.similarity_threshold
            ]

        candidates = documents[:candidate_k]
        rerank_scores: dict[str, float] = {}
        if reranker_on and candidates:
            rerank_started = time.perf_counter()
            candidates = get_compressor(s.reranker).compress_documents(candidates, question)[:top_k]
            timings["rerank"] = time.perf_counter() - rerank_started
            # LangChain's CrossEncoderReranker returns documents in rescored order but does not
            # attach the cross-encoder scores, so `rerank_score` is only set if a compressor
            # supplies relevance_score metadata; the fused score remains the reported score.
            rerank_scores = {
                document.metadata["chunk_id"]: float(document.metadata["relevance_score"])
                for document in candidates
                if "relevance_score" in document.metadata
            }
        else:
            candidates = candidates[:top_k]

        results: list[RetrievedChunk] = []
        for rank, document in enumerate(candidates, start=1):
            chunk = to_chunk(document)
            rerank_score = rerank_scores.get(chunk.chunk_id)
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=float(rerank_score if rerank_score is not None else scores.get(chunk.chunk_id, 0.0)),
                    rank=rank,
                    retrieval_method=strategy,  # type: ignore[arg-type]
                    fused_score=scores.get(chunk.chunk_id) if strategy == "hybrid" else None,
                    rerank_score=rerank_score,
                )
            )

        self.last_timings = timings
        return results

    def describe(self) -> dict:
        s = self.settings
        return {
            "embedding_model": s.embedding.model,
            "chunks": len(self.chunks),
            "strategy": s.retrieval.strategy,
            "top_k": s.retrieval.top_k,
            "candidate_k": s.retrieval.candidate_k,
            "bm25": s.bm25.enabled,
            "hybrid_fusion": s.hybrid.fusion if s.hybrid.enabled else None,
            "reranker": s.reranker.model if s.reranker.enabled else None,
            "index_cache": str(Path(self.cache.path)),
        }
