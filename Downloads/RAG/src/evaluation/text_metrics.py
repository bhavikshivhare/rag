"""Deterministic generation-quality metrics.

These are *not* LLM judgments and the report labels them as such:

* ``faithfulness``      - share of answer sentences whose best cosine similarity
                          against any retrieved context sentence is >= 0.55
                          (an entailment proxy, not true NLI).
* ``answer_relevancy``  - cosine similarity between question and answer embeddings.
* ``correctness``       - 0.7 * cosine(answer, reference) + 0.3 * token F1.
* ``token_f1``          - SQuAD-style bag-of-tokens F1 against the reference answer.

Unanswerable questions are scored by an explicit rule instead of an embedding
similarity: a refusal is correct, a fabrication is incorrect.
"""

from __future__ import annotations

import re
import string
from typing import Iterable

import numpy as np

from src.core.config import EvaluationConfig
from src.core.schemas import GenerationMetrics

SUPPORT_SIMILARITY = 0.55
# Refusal detection is deliberately narrow: a policy answer such as "add-ons are not
# included" must not be mistaken for a refusal. Patterns either anchor at the start of
# the answer or require the explicit "in the provided documents/context" phrasing.
REFUSAL_PATTERNS = [
    r"^\s*i (?:could not|couldn't|cannot|can't) (?:find|answer|determine)",
    r"^\s*i (?:do not|don't) know",
    r"^\s*(?:this|that|the) information (?:is )?not (?:available|provided)",
    r"could not find (?:this|that|the|any) information",
    r"no (?:information|details?) (?:is |are )?(?:available|provided|found)",
    r"not (?:available|mentioned|stated|specified|provided) in the (?:provided|supplied|given) (?:documents?|context)",
    r"(?:does not|doesn't) (?:contain|mention|specify|provide) (?:this|that|the) information",
    r"unable to (?:find|determine|answer) (?:this|that|the)",
    r"^\s*i'?m sorry, but i (?:cannot|can't|can not|am unable)",
    r"(?:cannot|can't|can not) (?:provide|answer|share|give) (?:answers?|information|details)?\s*(?:outside|beyond)",
]

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_SPLIT_RE.split(text or "") if len(part.strip()) > 3]


def normalize_answer(text: str) -> str:
    lowered = (text or "").lower().translate(PUNCT_TABLE)
    return " ".join(ARTICLES_RE.sub(" ", lowered).split())


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    from collections import Counter

    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype="float32").ravel()
    b = np.asarray(b, dtype="float32").ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def is_refusal(answer: str) -> bool:
    lowered = (answer or "").lower()
    return any(re.search(pattern, lowered) for pattern in REFUSAL_PATTERNS)


def _max_similarity_to_sentences(sentence_vector: np.ndarray, context_vectors: np.ndarray) -> float:
    if context_vectors.size == 0:
        return 0.0
    norms = np.linalg.norm(context_vectors, axis=1) * np.linalg.norm(sentence_vector)
    norms[norms == 0] = 1e-9
    return float(np.max(context_vectors @ sentence_vector / norms))


def compute_generation_metrics(
    question: str,
    answer: str,
    contexts: Iterable[str],
    ground_truth: str,
    expected_context: list[str],
    embed,
    evaluation_cfg: EvaluationConfig,
) -> GenerationMetrics:
    """``embed`` must expose ``embed_texts_raw(list[str]) -> np.ndarray``."""
    contexts = [c for c in contexts if c and c.strip()]
    answer = (answer or "").strip()
    refusal = is_refusal(answer)

    # ------------------------------------------------- unanswerable questions
    if not expected_context:
        return GenerationMetrics(
            faithfulness=1.0 if refusal else None,
            answer_relevancy=None,
            correctness=1.0 if refusal else 0.0,
            token_f1=None,
            unsupported_sentences=0,
            total_sentences=len(split_sentences(answer)),
            is_refusal=refusal,
        )

    if not answer:
        return GenerationMetrics(
            faithfulness=0.0 if contexts else None,
            answer_relevancy=0.0,
            correctness=0.0,
            token_f1=0.0,
            unsupported_sentences=0,
            total_sentences=0,
            is_refusal=False,
        )

    # ------------------------------------------------------- faithfulness
    answer_sentences = split_sentences(answer)
    context_sentences = [s for ctx in contexts for s in split_sentences(ctx)]
    faithfulness = None
    unsupported = 0
    if answer_sentences and context_sentences:
        vectors = embed.embed_texts_raw(answer_sentences + context_sentences)
        answer_vectors = vectors[: len(answer_sentences)]
        context_vectors = vectors[len(answer_sentences):]
        supported = 0
        for vector in answer_vectors:
            similarity = _max_similarity_to_sentences(vector, context_vectors)
            if similarity >= SUPPORT_SIMILARITY:
                supported += 1
        unsupported = len(answer_sentences) - supported
        faithfulness = supported / len(answer_sentences)

    # --------------------------------------------------- relevancy + correctness
    pair_vectors = embed.embed_texts_raw([question, answer, ground_truth])
    question_vector, answer_vector, reference_vector = pair_vectors[0], pair_vectors[1], pair_vectors[2]
    relevancy = cosine(question_vector, answer_vector)
    reference_similarity = cosine(answer_vector, reference_vector)
    f1 = token_f1(answer, ground_truth)
    correctness = max(0.0, min(1.0, 0.7 * reference_similarity + 0.3 * f1))

    return GenerationMetrics(
        faithfulness=faithfulness,
        answer_relevancy=max(0.0, min(1.0, relevancy)),
        correctness=correctness,
        token_f1=f1,
        unsupported_sentences=unsupported,
        total_sentences=len(answer_sentences),
        is_refusal=refusal,
    )
