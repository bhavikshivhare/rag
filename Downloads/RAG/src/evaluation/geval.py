"""G-Eval style LLM-as-a-judge scoring.

The judge is an LLM, not ground truth. It rates five criteria on a 1-5 rubric and
returns a short reason per criterion. Scores are normalised to 0-1 with
(score - 1) / 4. If the judge output cannot be parsed, the criterion keeps a
``None`` score and the error is reported - no default score is invented.
"""

from __future__ import annotations

import hashlib
import re

from src.core.config import JudgeConfig
from src.core.logging import get_logger
from src.core.schemas import GevalCriterionResult, GevalResult
from src.rag.llm import LLMError, get_llm

log = get_logger(__name__)

CRITERIA_DESCRIPTIONS = {
    "correctness": "factual agreement with the reference answer; wrong or missing facts score low",
    "completeness": "covers every part of the reference answer without omitting key details",
    "groundedness": "every claim is supported by the retrieved context; invented facts score low",
    "clarity": "clear, well organised, unambiguous wording",
    "conciseness": "no filler, repetition or irrelevant detail",
}

_LINE_RE = re.compile(r"(?im)^\s*[-*#>\s]*([a-z_ ]+?)\s*[:=\-]\s*([1-5](?:\.\d+)?)\s*(?:\|\s*(.*))?$")


def _build_prompt(question: str, answer: str, ground_truth: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(contexts)[:3000] if contexts else "(no context retrieved)"
    criteria_lines = "\n".join(f"- {name}: {desc}" for name, desc in CRITERIA_DESCRIPTIONS.items())
    return (
        "You are a strict evaluation judge for a retrieval-augmented QA system.\n"
        "Rate the ANSWER against the REFERENCE ANSWER and the CONTEXT on each criterion, "
        "using an integer from 1 (very poor) to 5 (excellent).\n\n"
        f"Criteria:\n{criteria_lines}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE ANSWER:\n{ground_truth}\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Return exactly one line per criterion, in this format, and nothing else:\n"
        "correctness: <1-5> | <short reason>\n"
        "completeness: <1-5> | <short reason>\n"
        "groundedness: <1-5> | <short reason>\n"
        "clarity: <1-5> | <short reason>\n"
        "conciseness: <1-5> | <short reason>"
    )


class GevalJudge:
    def __init__(self, cfg: JudgeConfig):
        self.cfg = cfg
        self.llm = get_llm(cfg, role="judge")
        self._cache: dict[str, GevalResult] = {}

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    def evaluate(self, question: str, answer: str, ground_truth: str, contexts: list[str]) -> GevalResult:
        if not self.cfg.enabled:
            return GevalResult(judge_model=self.cfg.model, error_message="judge disabled in configuration")
        if not answer or not answer.strip():
            return GevalResult(judge_model=self.cfg.model, error_message="empty answer - nothing to judge")

        cache_key = hashlib.sha1(
            "|".join([question, answer, ground_truth, str(len(contexts))]).encode("utf-8")
        ).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        messages = [
            {"role": "system", "content": "You are a precise, terse evaluation judge. Output only the requested lines."},
            {"role": "user", "content": _build_prompt(question, answer, ground_truth, contexts)},
        ]
        try:
            raw = self.llm.generate(messages, max_new_tokens=self.cfg.max_new_tokens, temperature=0.0).text
        except LLMError as exc:
            return GevalResult(judge_model=self.cfg.model, error_message=f"judge call failed: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("judge failed")
            return GevalResult(judge_model=self.cfg.model, error_message=f"judge failed: {exc}")

        result = self._parse(raw, self.cfg.criteria)
        self._cache[cache_key] = result
        return result

    @staticmethod
    def _parse(raw: str, expected_criteria: list[str]) -> GevalResult:
        scores: dict[str, tuple[float, str | None]] = {}
        for line in raw.splitlines():
            match = _LINE_RE.match(line.strip())
            if not match:
                continue
            name = match.group(1).strip().lower().replace("-", "_")
            if " " in name:
                continue
            if name not in CRITERIA_DESCRIPTIONS:
                continue
            score = float(match.group(2))
            scores[name] = (max(0.0, min(1.0, (score - 1) / 4)), (match.group(3) or "").strip() or None)

        criteria = [
            GevalCriterionResult(
                criterion=name,
                score=scores[name][0] if name in scores else None,
                reason=scores[name][1] if name in scores else None,
                raw_output=raw if name not in scores else None,
            )
            for name in expected_criteria
        ]
        missing = [name for name in expected_criteria if name not in scores]
        error = f"judge output missing criteria: {missing}" if len(missing) > len(expected_criteria) // 2 else None
        if not scores:
            error = "judge output could not be parsed into any criterion score"
        return GevalResult(judge_model="", criteria=criteria, error_message=error)
