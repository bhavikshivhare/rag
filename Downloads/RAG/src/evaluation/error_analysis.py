"""Rule-based error classification for failed or weak answers.

Classes: Retrieval Failure, Generation Failure, Hallucination, Insufficient
Context, Evaluation Failure, Other. Classification is deterministic and documented
so it can be audited; the rule order is the precedence order.
"""

from __future__ import annotations

from src.core.schemas import ErrorAnalysis, GenerationMetrics, RetrievalMetrics


def classify_error(
    *,
    error: str | None,
    expected_context: list[str],
    retrieval: RetrievalMetrics,
    generation: GenerationMetrics,
    overall_score: float | None,
    thresholds,
    is_refusal: bool,
) -> ErrorAnalysis:
    # 1. The run itself failed.
    if error:
        if error.startswith("retrieval failed"):
            return ErrorAnalysis(category="retrieval_failure", explanation=error, failed=True)
        return ErrorAnalysis(category="generation_failure", explanation=error, failed=True)

    answerable = bool(expected_context)

    # 2. Nothing relevant was retrieved, so the generator never had a chance.
    if answerable and retrieval.context_recall == 0.0:
        return ErrorAnalysis(
            category="retrieval_failure",
            explanation=(
                f"None of the expected documents {expected_context} appeared in the retrieved chunks, so the "
                "generator never had the supporting context."
            ),
            failed=True,
        )

    # 3. Unanswerable question answered as if it were answerable.
    if not answerable and not is_refusal:
        return ErrorAnalysis(
            category="hallucination",
            explanation="Question is unanswerable from the corpus but the model answered instead of refusing.",
            failed=True,
        )

    # 4. Answerable question, model refused anyway.
    if answerable and is_refusal:
        return ErrorAnalysis(
            category="insufficient_context",
            explanation="The model refused although the corpus contains the answer (often a partial retrieval miss).",
            failed=True,
        )

    # 5. Claims not supported by the retrieved context.
    faithfulness_threshold = thresholds.faithfulness
    if generation.faithfulness is not None and generation.faithfulness < faithfulness_threshold:
        return ErrorAnalysis(
            category="hallucination",
            explanation=(
                f"Only {generation.faithfulness:.0%} of answer sentences were supported by the retrieved context "
                f"({generation.unsupported_sentences}/{generation.total_sentences} unsupported)."
            ),
            failed=True,
        )

    # 6. Relevant context, but a wrong or incomplete answer.
    if overall_score is not None and overall_score < thresholds.overall_pass:
        return ErrorAnalysis(
            category="generation_failure",
            explanation=(
                "Relevant context was retrieved but the weighted score "
                f"{overall_score:.2f} is below the {thresholds.overall_pass:.2f} pass threshold."
            ),
            failed=True,
        )

    # 7. Only the LLM judge objected: treat as an evaluation signal, not an answer failure.
    if generation.correctness is None and generation.answer_relevancy is None:
        return ErrorAnalysis(
            category="evaluation_failure",
            explanation="No deterministic generation score could be computed for this item.",
            failed=True,
        )

    return ErrorAnalysis(category="none", explanation="", failed=False)
