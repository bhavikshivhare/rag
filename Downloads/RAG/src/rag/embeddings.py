"""Embedding provider: ``HuggingFaceEmbeddings`` with the BGE query instruction.

LangChain's embedding interface is what the vector store consumes, so the wrapper only
adds the two things this project needs: the query-side instruction prefix and a numpy
helper used by the similarity-based metrics.
"""

from __future__ import annotations

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from src.core.config import PROJECT_ROOT, EmbeddingConfig


class InstructionEmbeddings(Embeddings):
    """Hugging Face sentence-transformers through LangChain, instruction-aware on queries."""

    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self._model = HuggingFaceEmbeddings(
            model_name=cfg.model,
            cache_folder=str(PROJECT_ROOT / cfg.cache_dir),
            model_kwargs={"device": cfg.device},
            encode_kwargs={"normalize_embeddings": cfg.normalize, "batch_size": cfg.batch_size},
            show_progress=False,
        )

    @property
    def name(self) -> str:
        return self.cfg.model

    def embed_documents(self, texts) -> list[list[float]]:
        return self._model.embed_documents(list(texts))

    def embed_query(self, text: str) -> list[float]:
        query = f"{self.cfg.query_instruction}{text}" if self.cfg.query_instruction else text
        return self._model.embed_query(query)

    def embed_texts_raw(self, texts: list[str]) -> np.ndarray:
        """Embedding without the query instruction (used for similarity scoring)."""
        return np.asarray(self.embed_documents(texts), dtype="float32")


_MODEL_CACHE: dict[str, InstructionEmbeddings] = {}


def get_embedding_model(cfg: EmbeddingConfig) -> InstructionEmbeddings:
    key = f"{cfg.model}|{cfg.device}|{cfg.normalize}|{cfg.batch_size}"
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = InstructionEmbeddings(cfg)
    return _MODEL_CACHE[key]
