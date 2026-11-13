from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.core.config import HybridConfig
from src.rag.hybrid import make_hybrid_retriever, weighted_score_fusion


class StubRetriever(BaseRetriever):
    """Returns a fixed ranking, standing in for a real retriever."""

    documents: list[Document]

    def _get_relevant_documents(self, query, *, run_manager=None):
        return self.documents


def _document(chunk_id: str) -> Document:
    return Document(page_content=f"content {chunk_id}", metadata={"chunk_id": chunk_id})


def _ids(documents: list[Document]) -> list[str]:
    return [document.metadata["chunk_id"] for document in documents]


def test_ensemble_rewards_agreement_between_retrievers():
    vector = StubRetriever(documents=[_document("a#1"), _document("b#1")])
    bm25 = StubRetriever(documents=[_document("b#1"), _document("c#1")])
    retriever = make_hybrid_retriever(vector, bm25, HybridConfig(enabled=True, fusion="rrf", rrf_k=60))
    fused = retriever.invoke("question")
    assert _ids(fused)[0] == "b#1"
    assert set(_ids(fused)) == {"a#1", "b#1", "c#1"}


def test_ensemble_respects_weights():
    vector = StubRetriever(documents=[_document("a#1")])
    bm25 = StubRetriever(documents=[_document("b#1")])
    retriever = make_hybrid_retriever(
        vector, bm25, HybridConfig(enabled=True, fusion="rrf", vector_weight=1.0, bm25_weight=0.1)
    )
    assert _ids(retriever.invoke("question"))[0] == "a#1"


def test_weighted_fusion_normalises_scales():
    vector = [(_document("a#1"), 0.95), (_document("b#1"), 0.90)]
    bm25 = [(_document("b#1"), 30.0), (_document("a#1"), 1.0)]
    fused = weighted_score_fusion(vector, bm25, {"vector": 0.6, "bm25": 0.4})
    assert len(fused) == 2
    assert fused[0][0].metadata["chunk_id"] in {"a#1", "b#1"}
    assert fused[0][1] >= fused[1][1]
