# Metric Notes — formulas with one worked example

Every number below is the **real output of EXP-001** for question **Q021** (stored in
`reports/experiments/EXP-001_results.json`). Nothing here is illustrative or rounded into place —
each intermediate value was recomputed from the stored answer, contexts and reference, and matches
the stored metric.

Read this with `EXPLANATION.md` §6 (the evaluation workflow); this file is the arithmetic behind it.

---

## 0. The data for the example

| Field | Value |
| --- | --- |
| id | `Q021` |
| category / difficulty | multi-hop / hard |
| question | "What caused the April 2026 incident and what changes did Northwind make afterwards?" |
| expected_context | `["incident_postmortem_2026_04"]` |
| retrieved chunk ids (top 5) | `incident_postmortem_2026_04#506bf3d318`, `pricing#d1b8ca0a01`, `support_sla#ae8666bf5f`, `incident_postmortem_2026_04#eddc8f677d`, `onboarding_migration#9c2210055e` |
| answer | "The April 2026 incident was caused by a 47-minute outage that occurred on 2026-04-12. This led to approximately 3.4% of requests failing, resulting in no customer data loss or corruption. After the incident, Northwind made several improvements: 1. Increased Communication … 2. Improved Customer Support … 3. Enhanced Data Validation …" |
| ground truth | "The incident was caused by database connection pool exhaustion from a misconfigured autoscaler; Northwind raised pool limits, introduced canary deployments, and tightened alert thresholds." |

The answer reads fluently and cites real facts from the postmortem, but it answers the *wrong*
aspects (outage duration, request failure rate, communication) and never names the connection-pool
cause or the canary deployments. That is exactly the kind of failure the correctness metric exists to
catch, and it is the story the arithmetic below tells.

---

## 1. Retrieval metrics

### 1.1 Collapse chunks to documents

Relevance is defined per **document**, not per chunk. The 5 chunks become a ranked, deduplicated
document list (first occurrence keeps its position):

| rank | chunk | doc id | relevant? |
| ---: | --- | --- | :---: |
| 1 | `…#506bf3d318` | `incident_postmortem_2026_04` | 1 |
| 2 | `pricing#d1b8ca0a01` | `pricing` | 0 |
| 3 | `support_sla#ae8666bf5f` | `support_sla` | 0 |
| 4 | `…#eddc8f677d` | `incident_postmortem_2026_04` (duplicate) | — dropped |
| 5 | `onboarding_migration#9c2210055e` | `onboarding_migration` | 0 |

Ranked relevance vector: **`[1, 0, 0, 0]`**, retrieved doc count `K' = 4`,
expected set = {`incident_postmortem_2026_04`}, so `|expected| = 1`.

The dedup step matters: chunk-level it would look like 2 relevant out of 5 (0.40), document-level it
is 1 of 4 (0.25). The report always uses the document-level number.

### 1.2 The six retrieval metrics on this vector

| Metric | Formula | Substitution | Result |
| --- | --- | --- | ---: |
| Hit | any relevant retrieved | `sum([1,0,0,0]) = 1 > 0` | **1.0** |
| Recall@1 | `|expected ∩ docs[:1]| / |expected|` | `1/1` | **1.0** |
| Recall@3 | `|expected ∩ docs[:3]| / |expected|` | `1/1` | **1.0** |
| Recall@5 | `|expected ∩ docs[:5]| / |expected|` | `1/1` | **1.0** |
| Precision@K | `relevant / K'` | `1 / 4` | **0.25** |
| MRR | `1 / rank(first relevant)` | `1/1` | **1.0** |
| Context recall | `relevant_retrieved / |expected|` | `1/1` | **1.0** |
| Context precision | `Σ precision@i for relevant i, divided by min(K', |expected|)` | `(1/1) / min(4,1)` | **1.0** |

Two consequences worth internalising:

- **Hit rate and MRR are blind to the rest of the list.** This answer is wrong, yet retrieval scores
  1.0 — because the right document was found first. Retrieval metrics say "the corpus had what was
  needed", nothing more.
- **Context precision = 1.0 with 3 irrelevant documents** because its denominator is
  `min(K', |expected|) = 1`: only the first `|expected|` positions can be scored. Precision@K is the
  metric that punishes the noise (0.25). Do not read context precision as "half the context was junk".

For a multi-expected set, Q019 shows the same machinery with `|expected| = 2`:
docs `[security_compliance, support_sla, pricing]`, vector `[1,1,0]` → Recall@1 = `1/2` = **0.5**,
Precision@K = `2/3` = **0.667**, context precision = `(1/1 + 2/2)/min(3,2)` = **1.0**,
context recall = `2/2` = **1.0**.

---

## 2. Generation metrics

### 2.1 Token F1 — 0.0976

Normalise both strings: lowercase, strip all punctuation, drop `a/an/the`, split on whitespace.
Multiset overlap:

| Quantity | Value |
| --- | ---: |
| answer tokens `|P|` | 101 |
| reference tokens `|R|` | 22 |
| shared multiset `|P ∩ R|` | 6 → `and, by, caused, incident, northwind, was` |

```
precision = 6 / 101 = 0.0594
recall    = 6 / 22  = 0.2727
F1        = 2 · 0.0594 · 0.2727 / (0.0594 + 0.2727) = 0.0976
```

Sentence-level fluency earns nothing here: the six shared tokens are the question's own nouns plus
"was/by/caused". The content words that carry the answer — *database, connection, pool, exhaustion,
autoscaler, canary* — are absent, which is why F1 collapses to ~0.10 on a long, plausible answer.

### 2.2 Faithfulness — 1.0

Answer is split into **sentences**; each sentence is embedded and compared with every context
sentence; a sentence counts as supported when its best cosine similarity ≥ `0.55`.

| sentence | best cosine to a context sentence | supported? |
| ---: | ---: | :---: |
| 1 | 0.815 | yes |
| 2 | 0.808 | yes |
| 3 | 0.706 | yes |
| 4 | 0.861 | yes |
| 5 | 0.915 | yes |
| 6 | 0.887 | yes |
| 7 | 0.609 | yes |

```
faithfulness = supported_sentences / total_sentences = 7 / 7 = 1.0     (0 unsupported)
```

**This is the most important thing to understand about the whole benchmark.** Faithfulness is 1.0
because every sentence paraphrases *something* in the retrieved postmortem — the model is grounded,
not hallucinating. It is still the wrong answer. Faithfulness measures "did it make things up", not
"did it answer the question". A grounded, irrelevant answer scores 1.0 here and 0.51 on correctness.

The 0.55 threshold is lenient: sentence 7 sits at 0.609. At a hypothetical 0.85 the same answer would
score 3/7 = 0.43 and be classified as a hallucination — which would also be wrong, since those
sentences *are* supported. The threshold is a tunable in `text_metrics.SUPPORT_SIMILARITY`, not a
constant of nature.

### 2.3 Answer relevancy — 0.8108

```
cos(embed(question), embed(answer)) = 0.8108      clipped to [0, 1]
```

High because the answer is about Northwind's April incident. Note it stays high for a confident wrong
answer; it only measures topical alignment.

### 2.4 Correctness — 0.5103

```
reference_similarity = cos(embed(answer), embed(ground_truth)) = 0.6872
token_f1                                                      = 0.0976

correctness = 0.7 · 0.6872 + 0.3 · 0.0976
            = 0.48104 + 0.02928
            = 0.5103
```

The 70/30 blend is deliberate: embedding similarity is lenient (it rewards "same topic, same tone"),
token F1 is strict (it rewards the actual facts). A long, plausible, factually empty answer lands in
the middle — 0.51 — which is a fair description of it.

---

## 3. Weighted overall score — 0.8887

Only components that exist are used; the weights are renormalised over what is present (here all five
are present, so the sum is 1.0).

| Component | Value | Weight | Contribution |
| --- | ---: | ---: | ---: |
| Faithfulness | 1.0000 | 0.30 | 0.30000 |
| Answer relevancy | 0.8108 | 0.20 | 0.16216 |
| Context recall | 1.0000 | 0.20 | 0.20000 |
| Context precision | 1.0000 | 0.15 | 0.15000 |
| Correctness | 0.5103 | 0.15 | 0.07655 |
| **Total** | | **1.00** | **0.88870** |

```
overall = (0.30·1.0 + 0.20·0.8108 + 0.20·1.0 + 0.15·1.0 + 0.15·0.5103) / 1.00 = 0.8887
```

Note how retrieval's four perfect scores hold the total up: 65% of the weight is satisfiable by
finding the right document, whether or not the answer is right. That is a property of the rubric as
specified, not a bug — it is why correctness carries weight in the pass test too.

---

## 4. Thresholds — why this question fails

`passed` requires overall ≥ 0.60 **and** every per-metric threshold:

| Check | Value | Threshold | Result |
| --- | ---: | ---: | :---: |
| Overall | 0.8887 | 0.60 | pass |
| Faithfulness | 1.0000 | 0.60 | pass |
| Answer relevancy | 0.8108 | 0.55 | pass |
| Correctness | 0.5103 | 0.55 | **fail** |

One metric below its floor fails the question. This is the honest reading of Q021: 0.89 overall,
still a failure, because the answer does not contain the facts it was asked for.

### 4.1 A known gap, stated plainly

Q021 is recorded with `error_analysis.category = "none"` and `failed = False` even though `passed`
is `False`. `classify_error` tests the **overall** score against `overall_pass` (0.8887 ≥ 0.60, fine)
but has no branch for "a per-metric threshold failed". Consequences, visible in the text report:

- it **is** counted in `pass_rate` (0.92, i.e. 2 of 25 fail: Q021 and Q025);
- the report's Observations section names it as
  `Q021 (below threshold, no error class)`, because the classifier left the category as `none`;
- Q025 is the only question of the two that carries a real error class (`hallucination`).

Any comparison of "questions that failed thresholds" against "questions with an error class" will
disagree by exactly these cases. Fixing it means adding a branch to `classify_error` for per-metric
threshold breaches; the pass rate is unaffected either way.

---

## 5. Special cases (unanswerable questions)

`expected_context == []` switches off the reference-based metrics and replaces them with a
refusal test, because there is no correct answer to compare against.

| | Q024 (refused) | Q025 (answered anyway) |
| --- | --- | --- |
| answer | "I could not find this information in the provided documents." | a confident answer about a mobile app |
| is_refusal | True | False |
| faithfulness | 1.0 | `None` (no reference exists — not 1.0, not 0) |
| answer relevancy | `None` | `None` |
| token F1 | `None` | `None` |
| correctness | 1.0 | 0.0 |
| retrieval | `evaluable = False` (excluded from retrieval means) | same |
| overall | 1.0 (weights renormalised over 2 components) | 0.0 |
| error class | `none` (refusal is success) | `hallucination` |

Two deliberate choices visible here:

- Missing components are **dropped and the weights renormalised**, never filled with 0 or 1. Q024's
  score is `(0.30·1.0 + 0.15·1.0) / 0.45 = 1.0`, not a five-component average diluted by three zeros.
- Q025's `faithfulness = None` is what makes its hallucination classification come from the
  unanswerable-plus-answered rule rather than a low faithfulness score. The report prints `n/a`, which
  is accurate: there was no context to check the claim against, because the corpus never had the answer.

---

## 6. One-line summary of each metric's weakness

| Metric | High value means | What it does **not** mean |
| --- | --- | --- |
| Hit / Recall@K / MRR | the right document was found (early) | the answer used it |
| Precision@K | little irrelevant context | the relevant chunks were the useful ones |
| Context precision | the top of the list is relevant | the rest of the list is fine |
| Faithfulness | every sentence is grounded in context | the answer is correct |
| Answer relevancy | answer is on-topic | answer is right |
| Token F1 | the reference's words appear | they appear with the right meaning |
| Correctness | answer resembles the reference | the reference is exhaustive |
| G-Eval | a 0.5B model liked it | a human would agree |
