"""Cross-encoder reranking through LangChain's ``CrossEncoderReranker``.

The compressor rescores ``(question, chunk)`` pairs jointly and keeps ``top_n``; the
cross-encoder score lands in each document's ``relevance_score`` metadata, which is what
the retrieval layer reports.
"""

from __future__ import annotations

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

from src.core.config import PROJECT_ROOT, RerankerConfig

_COMPRESSORS: dict[str, CrossEncoderReranker] = {}


def get_compressor(cfg: RerankerConfig) -> CrossEncoderReranker:
    if cfg.model not in _COMPRESSORS:
        encoder = HuggingFaceCrossEncoder(
            model_name=cfg.model,
            model_kwargs={
                "device": "cpu",
                "max_length": 512,
                "cache_dir": str(PROJECT_ROOT / ".cache/huggingface"),
            },
        )
        _COMPRESSORS[cfg.model] = CrossEncoderReranker(model=encoder, top_n=cfg.top_n)
    return _COMPRESSORS[cfg.model]
