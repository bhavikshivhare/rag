"""Configuration system: YAML-driven, validated with Pydantic.

Resolution order (later wins):
    1. configs/config.yaml
    2. configs/models.yaml        (model registry defaults)
    3. configs/evaluation.yaml    (metric weights, judge + thresholds)
    4. --config <experiment.yaml> (experiment overrides)
    5. environment variables referenced from .env (e.g. OPENAI_API_KEY)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from src.core.tracing import configure_tracing

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"


# ---------------------------------------------------------------------------
# .env support (no external dependency, secrets never logged)
# ---------------------------------------------------------------------------
def load_env_file(path: Path | None = None) -> None:
    """Load simple KEY=VALUE pairs into os.environ without overriding existing values."""
    env_path = path or PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# ---------------------------------------------------------------------------
# Section models
# ---------------------------------------------------------------------------
class AppConfig(BaseModel):
    name: str = "RAG-Bench"
    environment: str = "local"
    seed: int = 42


class ExperimentMeta(BaseModel):
    experiment_id: str = "EXP-001"
    name: str = "Baseline vector RAG"
    description: str = ""
    dataset_path: str = "data/datasets/benchmark_v1.jsonl"
    dataset_version: str = "v1"

    @field_validator("experiment_id")
    @classmethod
    def _validate_experiment_id(cls, value: str) -> str:
        if not value.startswith("EXP-"):
            raise ValueError("experiment_id must look like EXP-001")
        return value

    def to_payload(self, settings: "Settings") -> dict[str, Any]:
        """Reproducible snapshot of everything that affects retrieval/answers."""
        s = settings
        return {
            "embedding_model": s.embedding.model,
            "chunk_size": s.chunking.chunk_size,
            "chunk_overlap": s.chunking.chunk_overlap,
            "retrieval_strategy": s.retrieval.strategy,
            "top_k": s.retrieval.top_k,
            "candidate_k": s.retrieval.candidate_k,
            "reranker_enabled": s.reranker.enabled,
            "reranker_model": s.reranker.model if s.reranker.enabled else None,
            "reranker_top_n": s.reranker.top_n if s.reranker.enabled else None,
            "hybrid_fusion": s.hybrid.fusion if s.hybrid.enabled else None,
            "llm": s.generation.model,
            "temperature": s.generation.temperature,
            "prompt_version": s.generation.prompt_version,
        }


class PathsConfig(BaseModel):
    documents: str = "data/documents"
    datasets: str = "data/datasets"
    cache: str = "data/cache"
    reports: str = "reports"
    database: str = "data/ragbench.db"
    logs: str = "logs"

    def resolve(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else PROJECT_ROOT / p


class LoggingConfig(BaseModel):
    level: str = "INFO"
    dir: str = "logs"
    json_sink: bool = False


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 32
    normalize: bool = True
    device: str = "cpu"
    query_instruction: str = "Represent this sentence for searching relevant passages: "
    cache_dir: str = ".cache/huggingface"


class ChunkingConfig(BaseModel):
    strategy: Literal["recursive", "fixed"] = "recursive"
    chunk_size: int = 700
    chunk_overlap: int = 80

    @model_validator(mode="after")
    def _validate(self) -> "ChunkingConfig":
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        return self


class RetrievalConfig(BaseModel):
    strategy: Literal["vector", "bm25", "hybrid"] = "vector"
    top_k: int = 5
    candidate_k: int = 20
    similarity_threshold: float = 0.0

    @model_validator(mode="after")
    def _validate(self) -> "RetrievalConfig":
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be >= top_k")
        return self


class BM25Config(BaseModel):
    enabled: bool = False
    k1: float = 1.5
    b: float = 0.75


class HybridConfig(BaseModel):
    enabled: bool = False
    fusion: Literal["rrf", "weighted"] = "rrf"
    rrf_k: int = 60
    vector_weight: float = 0.6
    bm25_weight: float = 0.4

    @model_validator(mode="after")
    def _validate(self) -> "HybridConfig":
        if self.fusion == "weighted" and abs(self.vector_weight + self.bm25_weight) < 1e-9:
            raise ValueError("hybrid weights must not both be zero")
        return self


class RerankerConfig(BaseModel):
    enabled: bool = False
    model: str = "BAAI/bge-reranker-base"
    top_n: int = 5


class GenerationConfig(BaseModel):
    provider: Literal["hf_local", "openai_compatible"] = "hf_local"
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    temperature: float = 0.0
    max_new_tokens: int = 220
    context_chunks: int = 5
    prompt_version: str = "v1"
    device: str = "cpu"
    cache_dir: str = ".cache/huggingface"


class JudgeConfig(BaseModel):
    enabled: bool = True
    provider: Literal["hf_local", "openai_compatible"] = "hf_local"
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    temperature: float = 0.0
    max_new_tokens: int = 320
    criteria: list[str] = Field(
        default_factory=lambda: ["correctness", "completeness", "groundedness", "clarity", "conciseness"]
    )


class DeepEvalConfig(BaseModel):
    """DeepEval runs only when explicitly enabled *and* an LLM judge is configured."""

    enabled: bool = False
    judge_provider: Literal["local", "openai"] = "local"
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "faithfulness": 0.7,
            "answer_relevancy": 0.7,
            "contextual_relevancy": 0.7,
            "contextual_recall": 0.7,
        }
    )


class MetricWeights(BaseModel):
    faithfulness: float = 0.30
    answer_relevancy: float = 0.20
    context_recall: float = 0.20
    context_precision: float = 0.15
    correctness: float = 0.15

    @model_validator(mode="after")
    def _validate(self) -> "MetricWeights":
        total = self.faithfulness + self.answer_relevancy + self.context_recall + self.context_precision + self.correctness
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"metric weights must sum to 1.0, got {total:.4f}")
        if any(v < 0 for v in self.model_dump().values()):
            raise ValueError("metric weights must be non-negative")
        return self


class ThresholdsConfig(BaseModel):
    retrieval_hit_similarity: float = 0.35
    faithfulness: float = 0.6
    answer_relevancy: float = 0.55
    correctness: float = 0.55
    context_recall: float = 0.5
    overall_pass: float = 0.6


class EvaluationConfig(BaseModel):
    weights: MetricWeights = Field(default_factory=MetricWeights)
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    deepeval: DeepEvalConfig = Field(default_factory=DeepEvalConfig)
    reference_similarity: float = 0.65  # answer<->ground-truth cosine to consider "correct"


class ModelsConfig(BaseModel):
    embedding_models: list[str] = Field(
        default_factory=lambda: ["BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"]
    )
    reranker_models: list[str] = Field(default_factory=lambda: ["BAAI/bge-reranker-base"])
    llms: list[str] = Field(default_factory=lambda: ["Qwen/Qwen2.5-0.5B-Instruct"])


class TracingConfig(BaseModel):
    """LangSmith tracing. The API key is read from the environment, never from YAML."""

    enabled: bool = False
    project: str = "rag-bench"
    endpoint: str = "https://api.smith.langchain.com"
    api_key_env: str = "LANGCHAIN_API_KEY"


class APIConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class PathsBundle(BaseModel):
    documents: Path
    datasets: Path
    cache: Path
    reports: Path
    database: Path
    logs: Path


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------
class Settings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    experiment: ExperimentMeta = Field(default_factory=ExperimentMeta)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    bm25: BM25Config = Field(default_factory=BM25Config)
    hybrid: HybridConfig = Field(default_factory=HybridConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    @property
    def paths_resolved(self) -> PathsBundle:
        return PathsBundle(
            documents=self.paths.resolve(self.paths.documents),
            datasets=self.paths.resolve(self.paths.datasets),
            cache=self.paths.resolve(self.paths.cache),
            reports=self.paths.resolve(self.paths.reports),
            database=self.paths.resolve(self.paths.database),
            logs=self.paths.resolve(self.paths.logs),
        )

    def ensure_directories(self) -> None:
        for path in self.paths_resolved.model_dump().values():
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True) if p.suffix else p.mkdir(parents=True, exist_ok=True)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def load_settings(experiment_config: str | Path | None = None) -> Settings:
    """Load and merge every config layer into validated :class:`Settings`."""
    load_env_file()
    merged: dict[str, Any] = {}
    for name in ("config.yaml", "models.yaml", "evaluation.yaml"):
        merged = _deep_merge(merged, _read_yaml(DEFAULT_CONFIG_DIR / name))

    if experiment_config:
        exp_path = Path(experiment_config)
        if not exp_path.is_absolute():
            exp_path = PROJECT_ROOT / exp_path
        merged = _deep_merge(merged, _read_yaml(exp_path))

    settings = Settings.model_validate(merged)
    if settings.hybrid.enabled:
        settings.bm25.enabled = True
        if settings.retrieval.strategy == "vector":
            settings.retrieval.strategy = "hybrid"
    if settings.retrieval.strategy in ("bm25", "hybrid"):
        settings.bm25.enabled = True
    settings.ensure_directories()
    configure_tracing(settings)
    return settings



