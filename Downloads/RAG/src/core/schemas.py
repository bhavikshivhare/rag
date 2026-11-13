"""Pydantic models shared across the pipeline, evaluation and reporting layers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

RetrievalMethod = Literal["vector", "bm25", "hybrid"]
FailureCategory = Literal[
    "none",
    "retrieval_failure",
    "generation_failure",
    "hallucination",
    "insufficient_context",
    "evaluation_failure",
    "other",
]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DocumentChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    doc_id: str
    content: str
    source: str
    doc_type: str
    page: int | None = None
    section: str | None = None
    chunk_index: int = 0
    char_start: int = 0

    def citation(self) -> str:
        parts = [self.doc_id]
        if self.section:
            parts.append(self.section)
        if self.page:
            parts.append(f"p.{self.page}")
        return " / ".join(parts)


class RetrievedChunk(BaseModel):
    chunk: DocumentChunk
    score: float
    rank: int
    retrieval_method: RetrievalMethod = "vector"
    original_scores: dict[str, float] = Field(default_factory=dict)
    fused_score: float | None = None
    rerank_score: float | None = None


class RAGResponse(BaseModel):
    question_id: str
    question: str
    answer: str
    retrieved: list[RetrievedChunk]
    contexts: list[str]
    sources: list[str]
    llm: str
    prompt_version: str
    retrieval_strategy: str
    reranker_model: str | None = None
    latency_ms: float
    retrieval_latency_ms: float
    rerank_latency_ms: float
    generation_latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None


class RetrievalMetrics(BaseModel):
    evaluable: bool = True  # False when the question has no expected context (unanswerable)
    hit: bool = False
    hit_rate: float = 0.0
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    relevant_retrieved: int = 0
    expected_retrieved: int = 0
    expected_total: int = 0


class GenerationMetrics(BaseModel):
    """Deterministic (embedding/lexical) generation quality signals."""

    faithfulness: float | None = None
    answer_relevancy: float | None = None
    correctness: float | None = None
    token_f1: float | None = None
    unsupported_sentences: int = 0
    total_sentences: int = 0
    is_refusal: bool = False


class DeepEvalMetricResult(BaseModel):
    metric: str
    score: float | None = None
    threshold: float | None = None
    passed: bool | None = None
    reason: str | None = None
    evaluator: str = "deepeval"
    error_message: str | None = None


class GevalCriterionResult(BaseModel):
    criterion: str
    score: float | None = None
    reason: str | None = None
    raw_output: str | None = None


class GevalResult(BaseModel):
    judge_model: str
    timestamp: datetime = Field(default_factory=utcnow)
    criteria: list[GevalCriterionResult] = Field(default_factory=list)
    error_message: str | None = None

    def score_of(self, criterion: str) -> float | None:
        for item in self.criteria:
            if item.criterion.lower() == criterion.lower():
                return item.score
        return None


class ErrorAnalysis(BaseModel):
    category: FailureCategory = "none"
    explanation: str = ""
    failed: bool = False


class QuestionResult(BaseModel):
    question_id: str
    question: str
    ground_truth: str
    category: str
    difficulty: str
    expected_context: list[str]
    answer: str
    retrieval: RetrievalMetrics
    generation: GenerationMetrics
    deepeval: list[DeepEvalMetricResult] = Field(default_factory=list)
    geval: GevalResult | None = None
    error_analysis: ErrorAnalysis
    overall_score: float | None = None
    overall_components: dict[str, float] = Field(default_factory=dict)
    passed: bool = False
    latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    retrieved_ids: list[str] = Field(default_factory=list)
    retrieved_sources: list[str] = Field(default_factory=list)
    contexts: list[str] = Field(default_factory=list)
    prompt_version: str = "v1"
    llm: str = ""
    error: str | None = None

    def gev_val(self, criterion: str) -> float | None:
        return self.geval.score_of(criterion) if self.geval else None


class AggregateMetrics(BaseModel):
    total_questions: int = 0
    answered_questions: int = 0
    hit_rate: float = 0.0
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    correctness: float = 0.0
    token_f1: float = 0.0
    geval_correctness: float | None = None
    geval_completeness: float | None = None
    geval_groundedness: float | None = None
    geval_clarity: float | None = None
    geval_conciseness: float | None = None
    geval_coverage: float = 0.0
    deepeval_faithfulness: float | None = None
    deepeval_answer_relevancy: float | None = None
    deepeval_contextual_relevancy: float | None = None
    deepeval_contextual_recall: float | None = None
    overall_score: float = 0.0
    pass_rate: float = 0.0
    hallucination_rate: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    avg_retrieval_latency_ms: float = 0.0
    avg_rerank_latency_ms: float = 0.0
    avg_generation_latency_ms: float = 0.0
    failure_counts: dict[str, int] = Field(default_factory=dict)


class ExperimentSummary(BaseModel):
    experiment_id: str
    name: str
    description: str = ""
    dataset_id: str = ""
    dataset_version: str = "v1"
    num_questions: int = 0
    config: dict[str, Any] = Field(default_factory=dict)
    metrics: AggregateMetrics = Field(default_factory=AggregateMetrics)
    status: str = "completed"
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime = Field(default_factory=utcnow)
    report_path: str | None = None
    notes: list[str] = Field(default_factory=list)


class EvaluationDatasetItem(BaseModel):
    id: str
    question: str
    ground_truth: str
    expected_context: list[str] = Field(default_factory=list)
    category: str = "factual"
    difficulty: str = "medium"
