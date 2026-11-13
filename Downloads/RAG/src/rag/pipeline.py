"""End-to-end RAG chain: question -> retrieval -> prompt | chat model -> answer.

The prompt and the model are composed with LCEL, so ``self.chain`` is a LangChain
runnable and every call is traceable in LangSmith when tracing is enabled.
"""

from __future__ import annotations

import time

from src.core.config import Settings
from src.core.logging import get_logger
from src.core.schemas import RAGResponse
from src.rag.llm import get_chat_model
from src.rag.prompts import format_context, get_prompt
from src.rag.retriever import RetrieverManager

log = get_logger(__name__)


class RAGPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.retriever = RetrieverManager(settings)
        self.prompt_version = settings.generation.prompt_version
        self.chain = get_prompt(self.prompt_version) | get_chat_model(settings.generation)

    def answer(self, question: str, question_id: str = "") -> RAGResponse:
        """Answer one question. Retrieval/generation failures are recorded, not hidden."""
        started = time.perf_counter()

        retrieval_started = time.perf_counter()
        try:
            retrieved = self.retriever.retrieve(question)
        except Exception as exc:
            log.error("retrieval failed for {}: {}", question_id or question[:40], exc)
            return RAGResponse(
                question_id=question_id,
                question=question,
                answer="",
                retrieved=[],
                contexts=[],
                sources=[],
                llm=self.settings.generation.model,
                prompt_version=self.prompt_version,
                retrieval_strategy=self.settings.retrieval.strategy,
                reranker_model=None,
                latency_ms=(time.perf_counter() - started) * 1000,
                retrieval_latency_ms=0.0,
                rerank_latency_ms=0.0,
                generation_latency_ms=0.0,
                error=f"retrieval failed: {exc}",
            )

        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        rerank_ms = self.retriever.last_timings.get("rerank", 0.0) * 1000
        context_chunks = retrieved[: self.settings.generation.context_chunks]

        generation_started = time.perf_counter()
        error = None
        answer = ""
        input_tokens = output_tokens = None
        try:
            response = self.chain.invoke(
                {"context": format_context(context_chunks), "question": question}
            )
            content = response.content
            answer = content if isinstance(content, str) else "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
            answer = answer.strip()
            usage = getattr(response, "usage_metadata", None) or {}
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
        except Exception as exc:  # model crash, OOM, endpoint error, template error
            error = f"generation failed: {exc}"
            log.exception("generation failed for {}", question_id)
        generation_ms = (time.perf_counter() - generation_started) * 1000

        if not answer and not error:
            error = "generation produced an empty answer"

        return RAGResponse(
            question_id=question_id,
            question=question,
            answer=answer,
            retrieved=retrieved,
            contexts=[item.chunk.content for item in context_chunks],
            sources=[item.chunk.citation() for item in context_chunks],
            llm=self.settings.generation.model,
            prompt_version=self.prompt_version,
            retrieval_strategy=self.settings.retrieval.strategy,
            reranker_model=self.settings.reranker.model if self.settings.reranker.enabled else None,
            latency_ms=(time.perf_counter() - started) * 1000,
            retrieval_latency_ms=retrieval_ms,
            rerank_latency_ms=rerank_ms,
            generation_latency_ms=generation_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            error=error,
        )
