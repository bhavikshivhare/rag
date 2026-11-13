"""Chat model factory built on LangChain.

Two interchangeable backends:
  * ``hf_local``          - ``HuggingFacePipeline`` wrapped by ``ChatHuggingFace``
  * ``openai_compatible`` - ``ChatOpenAI`` pointed at any OpenAI-compatible endpoint

``get_llm`` keeps the small ``generate(messages) -> LLMResult`` adapter used by the
evaluation judge and the DeepEval bridge; everything below it is LangChain.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_openai import ChatOpenAI

from src.core.config import PROJECT_ROOT


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMError(RuntimeError):
    """Raised when the model cannot produce an answer (never swallowed into a fake score)."""


_MESSAGE_TYPES = {
    "system": SystemMessage,
    "user": HumanMessage,
    "human": HumanMessage,
    "assistant": AIMessage,
    "ai": AIMessage,
}

_CHAT_MODELS: dict[str, BaseChatModel] = {}


def to_lc_messages(messages: list[dict[str, str]]) -> list[BaseMessage]:
    """Convert ``{"role": ..., "content": ...}`` dicts into LangChain messages."""
    converted: list[BaseMessage] = []
    for message in messages:
        role = str(message.get("role", "user")).lower()
        factory = _MESSAGE_TYPES.get(role, HumanMessage)
        converted.append(factory(content=message["content"]))
    return converted


def get_chat_model(cfg, max_new_tokens: int | None = None, temperature: float | None = None) -> BaseChatModel:
    """Build (or reuse) a LangChain chat model from a generation/judge config section."""
    provider = getattr(cfg, "provider", "hf_local")
    tokens = int(max_new_tokens or getattr(cfg, "max_new_tokens", 256))
    temp = getattr(cfg, "temperature", 0.0) if temperature is None else temperature
    device = getattr(cfg, "device", "cpu")
    cache_dir = getattr(cfg, "cache_dir", ".cache/huggingface")
    key = f"{provider}|{cfg.model}|{device}|{tokens}|{temp}|{cache_dir}"
    if key in _CHAT_MODELS:
        return _CHAT_MODELS[key]

    if provider == "hf_local":
        pipeline_kwargs: dict = {"max_new_tokens": tokens, "return_full_text": False, "do_sample": temp > 0}
        if temp > 0:
            pipeline_kwargs.update({"temperature": temp, "top_p": 0.9})
        llm = HuggingFacePipeline.from_model_id(
            model_id=cfg.model,
            task="text-generation",
            device=0 if device.startswith("cuda") else -1,
            model_kwargs={
                "cache_dir": str(PROJECT_ROOT / cache_dir),
                "dtype": "float32",
                "low_cpu_mem_usage": True,
            },
            pipeline_kwargs=pipeline_kwargs,
        )
        model: BaseChatModel = ChatHuggingFace(llm=llm)
    elif provider == "openai_compatible":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set; export it or use generation.provider='hf_local'")
        model = ChatOpenAI(
            model=cfg.model,
            temperature=temp,
            max_tokens=tokens,
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
            api_key=api_key,
            timeout=120,
        )
    else:  # pragma: no cover - config validation already restricts the values
        raise LLMError(f"unsupported LLM provider: {provider}")

    _CHAT_MODELS[key] = model
    return model


class ChatModelClient:
    """Minimal ``generate()`` adapter over a LangChain chat model."""

    def __init__(self, cfg, role: str = "generation"):
        self.cfg = cfg
        self.role = role

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResult:
        model = get_chat_model(self.cfg, max_new_tokens, temperature)
        try:
            response = model.invoke(to_lc_messages(messages))
        except Exception as exc:  # surface as a domain error, never as a score
            raise LLMError(f"LLM request failed: {exc}") from exc

        content = response.content
        if isinstance(content, str):
            text = content
        else:
            text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        usage = getattr(response, "usage_metadata", None) or {}
        return LLMResult(
            text=text.strip(),
            model=getattr(self.cfg, "model", ""),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )


def get_llm(cfg, role: str = "generation") -> ChatModelClient:
    """Compatibility factory used by the judge, DeepEval bridge and the pipeline."""
    return ChatModelClient(cfg, role)
