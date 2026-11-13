from pathlib import Path

from src.core.config import ChunkingConfig
from src.rag.chunking import chunk_documents
from src.rag.ingestion import load_documents, load_markdown

DOCS = Path(__file__).resolve().parents[1] / "data" / "documents"


def test_markdown_sections_become_separate_documents(tmp_path):
    path = tmp_path / "policy.md"
    path.write_text(
        "# Policy\n\nIntro text.\n\n## Refunds\n\nRefunds take 7 days.\n\n## Backups\n\nBackups take 35 days.\n",
        encoding="utf-8",
    )
    documents = load_markdown(path)
    assert documents
    assert {doc.metadata["doc_id"] for doc in documents} == {"policy"}
    sections = [doc.metadata.get("section") for doc in documents]
    assert "Refunds" in sections and "Backups" in sections


def test_chunks_carry_doc_and_section_metadata(tmp_path):
    path = tmp_path / "data_retention.md"
    path.write_text(
        "# Retention\n\n## Backups\n\nEncrypted backups are retained for 35 days on a rolling basis.\n",
        encoding="utf-8",
    )
    chunks = chunk_documents(load_markdown(path), ChunkingConfig(chunk_size=700, chunk_overlap=80))
    assert chunks
    assert chunks[0].doc_id == "data_retention"
    assert chunks[0].section == "Backups"
    assert "Backups" in chunks[0].content
    assert chunks[0].chunk_id.startswith("data_retention#")


def test_chunk_size_is_respected():
    documents = load_documents(DOCS)
    cfg = ChunkingConfig(chunk_size=400, chunk_overlap=50)
    chunks = chunk_documents(documents, cfg)
    # the section heading is prepended after splitting, so allow one heading of slack
    assert max(len(chunk.content) for chunk in chunks) <= cfg.chunk_size + 120


def test_corpus_documents_are_loaded():
    documents = load_documents(DOCS)
    assert documents
    assert "markdown" in {doc.metadata["doc_type"] for doc in documents}
    # every corpus file contributes sections under one unique document id
    assert len({doc.metadata["doc_id"] for doc in documents}) == len(list(DOCS.glob("*.md")))
