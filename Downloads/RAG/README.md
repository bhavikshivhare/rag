# RAG-Bench

A self-contained RAG evaluation and benchmarking framework. It ingests a document corpus, answers
benchmark questions with a configurable retrieval-augmented pipeline, scores every answer with
retrieval metrics, deterministic generation metrics and an LLM-as-a-judge rubric, classifies failures,
and writes a **plain-text report** that the API serves back for reading.

It is a benchmarking tool, not a chatbot: the primary output is a report, not a chat window.

### Stack

Retrieval, generation and tracing run on **LangChain**: `TextLoader` / `PyPDFLoader` /
`Docx2txtLoader` and `MarkdownHeaderTextSplitter` for ingestion, `RecursiveCharacterTextSplitter`
for chunking, `HuggingFaceEmbeddings`, `FAISS`, `BM25Retriever`, `EnsembleRetriever` (reciprocal
rank fusion) and `CrossEncoderReranker` for retrieval, and `ChatHuggingFace` / `ChatOpenAI` behind an
LCEL chain (`prompt | model`) for generation. The evaluation metrics, SQLite persistence, FastAPI
service and text report stay plain Python, because they are this project's own logic rather than
framework glue. Optional **LangSmith** tracing is environment-driven: flip `tracing.enabled` in
`configs/config.yaml` and export `LANGCHAIN_API_KEY`; with no key the run is untraced.

---

## What is measured

| Group | Metrics | How |
| --- | --- | --- |
| Retrieval | Hit Rate, Recall@1/3/5/10, Precision@K, MRR, Context Precision, Context Recall | deterministic, document-level relevance labels |
| Generation | Faithfulness, Answer Relevancy, Correctness, Token F1 | deterministic embedding / token proxies |
| LLM judge | Correctness, Completeness, Groundedness, Clarity, Conciseness | G-Eval rubric, 1-5 normalised to 0-1 |
| Optional | Faithfulness, Answer Relevancy, Contextual Relevancy, Contextual Recall | DeepEval, only when enabled |
| Operational | mean / p50 / p95 / p99 latency, retrieval and rerank latency | measured per question |

The weighted overall score uses `configs/evaluation.yaml`:

```
0.30·faithfulness + 0.20·answer_relevancy + 0.20·context_recall + 0.15·context_precision + 0.15·correctness
```

Weights must sum to 1.0 (validated at load time). When a component genuinely cannot be computed for a
question, it is dropped and the remaining weights are renormalised. **No metric is ever filled with a
default or placeholder score.**

### Honesty notes that appear in the reports

- Faithfulness / relevancy / correctness are embedding- and token-based estimates, labelled
  "deterministic" in every table. They are not entailment judgements.
- G-Eval scores are labelled **LLM-as-a-judge** and include judge coverage (how many answers the
  judge actually scored).
- If the judge output cannot be parsed, the criterion is reported as `n/a` with an error, never 0.
- DeepEval results are absent unless explicitly enabled; the report says so rather than inventing them.
- Retrieval relevance is graded per document (`expected_context`), not per chunk.

---

## Quick start

```bash
py -3.10 -m venv .venv
.venv/Scripts/python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1
.venv/Scripts/python.exe -m pip install -r requirements.txt

.venv/Scripts/python.exe -m src.rag.ingestion
.venv/Scripts/python.exe -m src.evaluation.run_evaluation --config configs/experiments/exp_001_baseline.yaml
.venv/Scripts/python.exe -m src.evaluation.run_evaluation --config configs/experiments/exp_002_enhanced.yaml
```

A run writes `reports/experiments/EXP-00N_report.txt` - no PDF, HTML, Excel or DOCX is produced.
The stored baseline lives in `reports/baseline/` and is replaced only on an explicit request
(`POST /api/baseline`, or `--save-baseline` on the report CLI).

Full command reference, API examples, Docker and cache-reset commands: **[RUN_COMMANDS.md](RUN_COMMANDS.md)**.
How every file, module and workflow fits together: **[EXPLANATION.md](EXPLANATION.md)**.
Every metric's maths with a worked example from the real run: **[METRICS.md](METRICS.md)**.
Shortcut: `./run.sh all` (Git Bash) or `.\run.ps1 all` (PowerShell).

---

## The two benchmarked configurations

`configs/experiments/exp_001_baseline.yaml` — the "before" state:

- vector-only retrieval (FAISS, cosine), top 5
- prompt v1: plain "answer from the context", no grounding rule, no refusal instruction
- no BM25, no reranker

`configs/experiments/exp_002_enhanced.yaml` — the "after" state, three changes in one bundle:

1. **Hybrid retrieval** — BM25 + vector candidates fused with reciprocal rank fusion (k=60)
2. **Prompt v2** — strict grounding, source citations, and the exact refusal sentence
   `I could not find this information in the provided documents.`
3. **Reranker** — `BAAI/bge-reranker-base` cross-encoder over the top 20 candidates → final top 5

Everything else (corpus, chunks, embeddings, generator, questions) is identical, so any difference
is attributable to the bundle. It is **not** an ablation — isolating each change needs three extra runs.

---

## Corpus and dataset

`data/documents/` — 8 fictional Northwind Analytics Markdown documents covering pricing,
refunds, security, retention, support SLA, API limits, onboarding, integrations, release notes and an
incident postmortem.

`data/datasets/benchmark_v1.jsonl` — 23 questions across `factual`, `comparison`, `multi-hop`,
`summarization` and `unanswerable`, each with a reference answer and document-level `expected_context`.
Unanswerable questions have an empty `expected_context` and are excluded from retrieval aggregates.

---

## Layout

```
configs/            config.yaml, models.yaml, evaluation.yaml, experiments/*.yaml
data/documents/     source corpus (md, pdf, docx)
data/datasets/      benchmark_v1.jsonl
data/cache/index/   FAISS index cache, keyed by model + corpus fingerprint
src/core/           config, logging, Pydantic schemas
src/rag/            ingestion, chunking, embeddings, vector store, BM25, hybrid fusion,
                    reranker, prompts, LLM factory, pipeline
src/evaluation/     dataset, retrieval metrics, text metrics, G-Eval, DeepEval adapter,
                    error analysis, scoring, run_evaluation
src/evaluation/text_report.py  plain-text report builder + persistent baseline store
src/db/             SQLAlchemy models + SQLite repository
src/api/            FastAPI service
tests/              unit tests (no model downloads)
```

## Models

All model names live in configuration — nothing is hard-coded:

- embeddings: `BAAI/bge-small-en-v1.5` (alternative: `sentence-transformers/all-MiniLM-L6-v2`)
- reranker: `BAAI/bge-reranker-base`
- generator / judge: `Qwen/Qwen2.5-0.5B-Instruct` by default; any OpenAI-compatible endpoint can be
  used by setting `generation.provider: openai_compatible` and `OPENAI_API_KEY`.

## API

`python -m src.api.main` exposes dataset upload/listing, experiment and evaluation runs, per-question
results, text-report and metrics retrieval, baseline storage and a health check. See RUN_COMMANDS.md section 8.

## Limitations

- The benchmark has 23 questions; single-question changes move aggregate metrics by ~4 points.
- Reference answers are not exhaustive.
- The default local generator and judge are small models; scores are directional, not authoritative.
- Absolute latency numbers are CPU-specific.
