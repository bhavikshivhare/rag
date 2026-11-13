"""Document ingestion built on LangChain loaders and splitters.

Markdown goes through ``MarkdownHeaderTextSplitter`` so each H2 section becomes its own
document with ``section`` metadata; PDF pages and DOCX files come from the community
loaders. Every loaded document carries ``doc_id``, ``source`` and ``doc_type`` metadata,
which the chunker turns into chunk ids.

CLI:
    python -m src.rag.ingestion            # ingest + build the retrieval index
    python -m src.rag.ingestion --stats    # only print corpus statistics
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from src.core.config import load_settings
from src.core.logging import get_logger, setup_logging
from src.core.schemas import DocumentChunk
from src.rag.chunking import chunk_documents, slugify

log = get_logger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown", ".docx"}
MARKDOWN_HEADERS = [("#", "title"), ("##", "section"), ("###", "subsection")]


def _metadata(path: Path, doc_type: str) -> dict[str, str]:
    return {"doc_id": slugify(path.stem), "source": str(path), "doc_type": doc_type}


def load_markdown(path: Path) -> list[Document]:
    """Split a markdown file on its headings; one document per section."""
    text = "\n".join(doc.page_content for doc in TextLoader(str(path), encoding="utf-8").load())
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=MARKDOWN_HEADERS, strip_headers=True)
    meta = _metadata(path, "markdown")
    documents = []
    for part in splitter.split_text(text):
        metadata = {**meta, **part.metadata}
        documents.append(Document(page_content=part.page_content, metadata=metadata))
    return documents


def load_text(path: Path) -> list[Document]:
    documents = TextLoader(str(path), encoding="utf-8").load()
    for document in documents:
        document.metadata.update(_metadata(path, "text"))
    return documents


def load_pdf(path: Path) -> list[Document]:
    documents = PyPDFLoader(str(path)).load()
    for document in documents:
        # PyPDFLoader pages are zero-based; the schema and citations are one-based.
        document.metadata.update(_metadata(path, "pdf"))
        document.metadata["page"] = int(document.metadata.get("page", 0)) + 1
    return documents


def load_docx(path: Path) -> list[Document]:
    documents = Docx2txtLoader(str(path)).load()
    for document in documents:
        document.metadata.update(_metadata(path, "docx"))
    return documents


LOADERS = {
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".txt": load_text,
    ".pdf": load_pdf,
    ".docx": load_docx,
}


def load_documents(documents_dir: str | Path) -> list[Document]:
    directory = Path(documents_dir)
    if not directory.exists():
        raise FileNotFoundError(f"documents directory not found: {directory}")

    documents: list[Document] = []
    seen: dict[str, int] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            loaded = LOADERS[path.suffix.lower()](path)
        except Exception as exc:  # keep going; report the bad file
            log.error("failed to load {}: {}", path.name, exc)
            continue
        # Two files with the same stem must stay distinguishable, but every section of
        # one file shares that file's id.
        base = slugify(path.stem)
        seen[base] = seen.get(base, 0) + 1
        doc_id = base if seen[base] == 1 else f"{base}_{seen[base]}"
        for document in loaded:
            document.metadata["doc_id"] = doc_id
            documents.append(document)

    if not documents:
        raise ValueError(f"no supported documents in {directory}")
    return documents


def corpus_fingerprint(documents_dir: str | Path, chunk_cfg) -> str:
    """Fingerprint of the corpus + chunking config, used to key the index cache."""
    hasher = hashlib.sha256()
    hasher.update(f"chunk={chunk_cfg.chunk_size}:{chunk_cfg.chunk_overlap}:{chunk_cfg.strategy}".encode())
    for path in sorted(Path(documents_dir).rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            stat = path.stat()
            hasher.update(f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}".encode())
    return hasher.hexdigest()[:16]


def ingest(settings) -> list[DocumentChunk]:
    documents = load_documents(settings.paths_resolved.documents)
    chunks = chunk_documents(documents, settings.chunking)
    log.info(
        "ingested {} sections -> {} chunks (chunk_size={}, overlap={})",
        len(documents),
        len(chunks),
        settings.chunking.chunk_size,
        settings.chunking.chunk_overlap,
    )
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents and build the retrieval index")
    parser.add_argument("--config", default=None, help="optional experiment config override")
    parser.add_argument("--stats", action="store_true", help="print corpus statistics only")
    args = parser.parse_args()

    settings = load_settings(args.config)
    setup_logging(settings.paths_resolved.logs, settings.logging.level)

    chunks = ingest(settings)
    by_doc: dict[str, int] = {}
    for chunk in chunks:
        by_doc[chunk.doc_id] = by_doc.get(chunk.doc_id, 0) + 1

    print(json.dumps({"documents": by_doc, "total_chunks": len(chunks)}, indent=2))
    if not args.stats:
        from src.rag.retriever import RetrieverManager

        manager = RetrieverManager(settings)
        manager.ensure_index(chunks)
        print("index ready:", manager.cache.describe())


if __name__ == "__main__":
    main()
