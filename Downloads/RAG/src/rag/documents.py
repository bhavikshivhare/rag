"""Bridge between the project's chunk schema and LangChain ``Document`` objects.

Every LangChain component (vector store, BM25 retriever, ensemble, compressor) exchanges
``Document`` instances, while the evaluation layer works with :class:`DocumentChunk`.
Converting in one place keeps chunk ids, section and page metadata intact across hops.
"""

from __future__ import annotations

from langchain_core.documents import Document

from src.core.schemas import DocumentChunk

CHUNK_KEY = "chunk"
CHUNK_ID_KEY = "chunk_id"
DOC_ID_KEY = "doc_id"


def to_document(chunk: DocumentChunk) -> Document:
    """Wrap a chunk so it can travel through LangChain retrievers and compressors."""
    return Document(
        page_content=chunk.content,
        metadata={
            CHUNK_KEY: chunk.model_dump(),
            CHUNK_ID_KEY: chunk.chunk_id,
            DOC_ID_KEY: chunk.doc_id,
        },
    )


def to_chunk(document: Document) -> DocumentChunk:
    """Recover the original chunk from a LangChain document."""
    payload = document.metadata.get(CHUNK_KEY)
    if not isinstance(payload, dict):
        raise ValueError("document has no chunk payload - it was not built by to_document()")
    return DocumentChunk.model_validate(payload)
