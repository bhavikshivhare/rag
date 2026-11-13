"""Prompt templates as LangChain ``ChatPromptTemplate`` objects.

v1 is the deliberately plain baseline prompt: it asks for a helpful answer from the
context but never tells the model what to do when the context is incomplete.
v2 adds explicit grounding rules, citation of the source id, and a fixed refusal
sentence for unanswerable questions. The only difference between the two is the
instruction block, so any metric change is attributable to the prompt.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from src.core.schemas import DocumentChunk, RetrievedChunk

REFUSAL_MESSAGE = "I could not find this information in the provided documents."

# version -> (system instruction, human template)
PROMPT_TEXT: dict[str, tuple[str, str]] = {
    "v1": (
        "You are a helpful assistant for Northwind Analytics.",
        (
            "Answer the question using the context below.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        ),
    ),
    "v2": (
        "You are a customer support assistant for Northwind Analytics. You answer strictly "
        "from the supplied documents and never invent facts, numbers, dates or policies.",
        (
            "Answer the question using only the context below.\n"
            "Rules:\n"
            "- Reply with complete sentences that state the answer. Never reply with only a source id.\n"
            "- End each sentence with the source id in square brackets, like this: "
            '"Refunds are processed within 30 days [pricing]."\n'
            "- Do not use outside knowledge and do not guess.\n"
            "- If the context does not contain the answer, reply exactly: "
            f'"{REFUSAL_MESSAGE}"\n\n'
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        ),
    ),
}

_ROLES = {"system": "system", "human": "user", "ai": "assistant"}


def get_prompt(version: str) -> ChatPromptTemplate:
    if version not in PROMPT_TEXT:
        raise ValueError(f"unknown prompt version '{version}'; available: {sorted(PROMPT_TEXT)}")
    system, user = PROMPT_TEXT[version]
    return ChatPromptTemplate.from_messages([("system", system), ("human", user)])


def _as_chunk(item: RetrievedChunk | DocumentChunk) -> DocumentChunk:
    return item.chunk if isinstance(item, RetrievedChunk) else item


def format_context(chunks: list[RetrievedChunk] | list[DocumentChunk]) -> str:
    blocks: list[str] = []
    for item in chunks:
        chunk = _as_chunk(item)
        location = f" page {chunk.page}" if chunk.page else ""
        blocks.append(f"[{chunk.doc_id}{location}]\n{chunk.content}")
    return "\n\n---\n\n".join(blocks)


def build_messages(
    question: str, chunks: list[RetrievedChunk] | list[DocumentChunk], version: str
) -> list[dict[str, str]]:
    """Render the prompt as plain role/content dicts (used by tests and the judge)."""
    messages = get_prompt(version).format_messages(context=format_context(chunks), question=question)
    return [{"role": _ROLES.get(message.type, "user"), "content": str(message.content)} for message in messages]
