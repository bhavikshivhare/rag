"""Chunking built on LangChain's recursive splitter.

Chunks are produced from LangChain ``Document`` objects (one per structural section,
see :mod:`src.rag.ingestion`) so a chunk never silently spans two unrelated headings.
Each chunk gets a stable ``<doc_id>#<hash>`` identifier because the benchmark and the
retrieval metrics are keyed on those ids.
"""

from __future__ import annotations

import hashlib
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import ChunkingConfig
from src.core.schemas import DocumentChunk

SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "document"


def make_splitter(cfg: ChunkingConfig) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=SEPARATORS,
        keep_separator=True,
        length_function=len,
    )


def chunk_documents(documents: list[Document], cfg: ChunkingConfig) -> list[DocumentChunk]:
    """Split loaded documents into chunks, keeping doc/section/page metadata."""
    splitter = make_splitter(cfg)
    chunks: list[DocumentChunk] = []

    for document in documents:
        meta = document.metadata
        doc_id = str(meta.get("doc_id") or slugify(str(meta.get("source", "document"))))
        section = meta.get("section")
        pieces = [document] if cfg.strategy == "fixed" else splitter.split_documents([document])

        for index, piece in enumerate(pieces):
            body = piece.page_content.strip()
            if not body:
                continue
            # The heading is repeated inside the chunk so lexical (BM25) and vector
            # retrieval can both match on the section topic.
            content = f"{section}\n\n{body}" if section else body
            digest = hashlib.sha1(f"{doc_id}|{index}|{content}".encode("utf-8")).hexdigest()[:10]
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_id}#{digest}",
                    doc_id=doc_id,
                    content=content,
                    source=str(meta.get("source", "")),
                    doc_type=str(meta.get("doc_type", "text")),
                    page=meta.get("page"),
                    section=section,
                    chunk_index=index,
                    char_start=0,
                )
            )
    return chunks
