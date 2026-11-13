import pytest

from src.core.schemas import DocumentChunk, RetrievedChunk
from src.rag.prompts import REFUSAL_MESSAGE, build_messages, format_context, get_prompt


def _retrieved():
    chunk = DocumentChunk(chunk_id="pricing#1", doc_id="pricing", content="Growth costs $199 per month.",
                          source="pricing.md", doc_type="markdown", section="Plans")
    return [RetrievedChunk(chunk=chunk, score=0.9, rank=1)]


def test_v1_has_no_refusal_instruction():
    messages = build_messages("How much is Growth?", _retrieved(), "v1")
    assert "could not find" not in messages[1]["content"].lower()
    assert "Growth costs $199" in messages[1]["content"]


def test_v2_requires_grounding_and_refusal():
    messages = build_messages("How much is Growth?", _retrieved(), "v2")
    content = messages[1]["content"]
    assert REFUSAL_MESSAGE in content
    assert "only the context" in content.lower()
    assert "[pricing]" in format_context(_retrieved())


def test_unknown_prompt_version_is_rejected():
    with pytest.raises(ValueError):
        get_prompt("v99")
