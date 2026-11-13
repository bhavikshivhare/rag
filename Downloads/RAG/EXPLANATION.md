# RAG-Bench — How It Works

A technical walkthrough of every module, workflow and number in this repository. It describes the
code as it exists on disk, with measured values where the project has measurements and `N/A`
where it does not.

---

## 1. What the project does

RAG-Bench answers a fixed benchmark against a fixed document corpus and scores every answer, so a
change to retrieval or prompting can be judged by numbers instead of by reading a few outputs.

```
documents ──► chunk ──► index ──► retrieve ──► prompt ──► generate ──► score ──► report
                                                                        └──► metrics file
```

Two experiment configurations are shipped: a vector-only baseline (`EXP-001`) and a bundle of
hybrid retrieval, prompt v2 and a cross-encoder reranker (`EXP-002`). They differ only in their
experiment YAML, so any difference in results comes from those settings — not from the corpus,
chunker, embedding model, generator, dataset or metrics, which are shared.

**What this is not:** a chatbot, an ablation study, or a service that invents missing scores. A
metric that genuinely cannot be computed is reported as `N/A` and excluded from the weighted score,
whose remaining weights are renormalised.

---

## 2. Repository map

Line counts are actual (`wc -l`), and every module listed exists.

```
.
├── configs/
│   ├── config.yaml                runtime defaults: paths, chunking, retrieval, reranker, generation,
│   │                              tracing, api
│   ├── models.yaml                model registry (embedding / reranker / LLM names)
│   ├── evaluation.yaml            metric weights, thresholds, judge + DeepEval settings
│   └── experiments/
│       ├── exp_001_baseline.yaml  EXP-001: vector, prompt v1, no reranker
│       └── exp_002_enhanced.yaml  EXP-002: hybrid RRF, prompt v2, reranker
│
├── data/
│   ├── documents/                 8 Markdown documents (the corpus)
│   ├── datasets/benchmark_v1.jsonl  23 questions with ground truth + expected_context
│   ├── cache/index/<key>/         FAISS index cache (index.faiss, index.pkl, chunks.json)
│   └── ragbench.db                SQLite: datasets, experiments, results, reports
│
├── src/
│   ├── core/
│   │   ├── config.py              Pydantic settings + YAML deep-merge loader (356 lines)
│   │   ├── logging.py             loguru setup, one rotating file per day (36 lines)
│   │   ├── schemas.py             every data contract: chunks, responses, metrics (235 lines)
│   │   └── tracing.py             LangSmith env-var setup, off without a key (45 lines)
│   ├── rag/                       LangChain layer
│   │   ├── documents.py           Document ↔ DocumentChunk conversion (36 lines)
│   │   ├── ingestion.py           loaders, markdown section split, fingerprint (160 lines)
│   │   ├── chunking.py            recursive splitter + stable chunk ids (70 lines)
│   │   ├── embeddings.py          HuggingFaceEmbeddings + BGE query instruction (53 lines)
│   │   ├── vectorstore.py         FAISS store, save_local/load_local (84 lines)
│   │   ├── bm25.py                BM25Retriever + raw score lookup (54 lines)
│   │   ├── hybrid.py              EnsembleRetriever (RRF) + weighted fallback (63 lines)
│   │   ├── reranker.py            CrossEncoderReranker compressor (29 lines)
│   │   ├── prompts.py             ChatPromptTemplate v1 / v2 (76 lines)
│   │   ├── llm.py                 ChatHuggingFace / ChatOpenAI + adapter (138 lines)
│   │   ├── retriever.py           RetrieverManager + IndexCache (213 lines)
│   │   └── pipeline.py            RAGPipeline: retrieve → LCEL chain (101 lines)
│   ├── evaluation/
│   │   ├── dataset.py             JSONL validation + statistics (88 lines)
│   │   ├── retrieval_metrics.py   hit, recall@k, precision@k, MRR, context P/R (75 lines)
│   │   ├── text_metrics.py        faithfulness, relevancy, correctness, token F1 (166 lines)
│   │   ├── geval.py               LLM-as-a-judge rubric (119 lines)
│   │   ├── deepeval_adapter.py    optional DeepEval metrics (145 lines)
│   │   ├── scoring.py             weighted score, thresholds, aggregation (146 lines)
│   │   ├── error_analysis.py      failure classification rules (89 lines)
│   │   ├── run_evaluation.py      experiment orchestrator + CLI (315 lines)
│   │   └── text_report.py         report + metrics file + baseline store (505 lines)
│   ├── db/
│   │   ├── models.py              SQLAlchemy tables (85 lines)
│   │   ├── session.py             engine, session_scope, FastAPI dependency (66 lines)
│   │   └── repository.py          all reads/writes (138 lines)
│   └── api/
│       └── main.py                FastAPI service, background jobs (328 lines)
│
├── tests/                         40 unit tests, no model downloads
├── reports/
│   ├── experiments/EXP-00N_{report.txt, metrics.txt, results.json, summary.json}
│   └── baseline/{baseline.txt, baseline.json}     explicit saves only
│
├── run.sh / run.ps1               wrappers: setup|ingest|baseline|enhanced|reports|api|test|all
├── Dockerfile                     container image, runs the baseline by default
├── requirements.txt               pinned ranges
├── pytest.ini                     testpaths=tests
└── .env.example                   optional external LLM / LangSmith keys
```

---

## 3. Architecture at a glance

```
        configs/*.yaml ──deep-merge──► Settings (Pydantic, validated)
                                            │
              ┌─────────────────────────────┴─────────────────────────────┐
              ▼                                                           ▼
      run_evaluation (CLI / API job)                              src/api/main.py
              │                                                           │
              │ per question                                              │ reports / metrics
              ▼                                                           ▼
   ┌──────────────────────────────┐                            reports/experiments/*.txt
   │ RAGPipeline                  │
   │  ├── RetrieverManager        │──► LangChain FAISS + BM25Retriever
   │  │                           │      + EnsembleRetriever + CrossEncoderReranker
   │  └── chain: prompt | model   │──► ChatPromptTemplate + ChatHuggingFace
   └──────────────┬───────────────┘
                  ▼
   retrieval_metrics → text_metrics → GevalJudge → DeepEval (opt-in)
                  ▼
   scoring.weighted_score → error_analysis.classify_error → aggregate
                  ▼
   db.repository (SQLite)  +  text_report (report, metrics file, baseline)
```

---

## 4. The LangChain stack

Everything that retrieves, ranks or generates is a LangChain component; the evaluation, storage,
API and report layers are this project's own code.

| Step | Component | Configured by |
| --- | --- | --- |
| Load files | `TextLoader`, `PyPDFLoader`, `Docx2txtLoader` | file suffix |
| Section split | `MarkdownHeaderTextSplitter` (H1/H2/H3) | `MARKDOWN_HEADERS` |
| Chunk | `RecursiveCharacterTextSplitter` | `chunking.chunk_size` / `chunk_overlap` |
| Embed | `HuggingFaceEmbeddings` | `embedding.*` |
| Vector search | `FAISS` with `MAX_INNER_PRODUCT` | `retrieval.*` |
| Lexical search | `BM25Retriever` (rank_bm25 underneath) | `bm25.*` |
| Fusion | `EnsembleRetriever` (RRF); min-max weighted fallback | `hybrid.*` |
| Rerank | `CrossEncoderReranker` + `HuggingFaceCrossEncoder` | `reranker.*` |
| Prompt | `ChatPromptTemplate` (v1 / v2) | `generation.prompt_version` |
| Generate | `ChatHuggingFace` or `ChatOpenAI`, composed with LCEL | `generation.*` |
| Tracing | LangSmith (env vars only) | `tracing.*` |

`src/rag/documents.py` converts between LangChain `Document` objects and the project's
`DocumentChunk` schema in both directions. That boundary is why the evaluation layer never depends
on a LangChain internal: chunk ids, `doc_id`, `section` and `page` survive every hop.

---

## 5. Workflow A — ingestion and index build

**Entry points:** `python -m src.rag.ingestion`, or implicitly the first `RetrieverManager.ensure_index()`.

```
data/documents/*.md|txt|pdf|docx
        │
        │ load_documents(dir)                          ingestion.py
        ▼
  ┌────────────────────────────────────────────────────────────┐
  │ per suffix:                                                │
  │   .md   TextLoader → MarkdownHeaderTextSplitter            │
  │   .txt  TextLoader                                         │
  │   .pdf  PyPDFLoader (page numbers made 1-based)            │
  │   .docx Docx2txtLoader                                     │
  │ → Document(page_content, metadata={doc_id, source,         │
  │            doc_type, section?})                            │
  └───────────────────────────┬────────────────────────────────┘
                              │ chunk_documents(docs, chunking)   chunking.py
                              ▼
  RecursiveCharacterTextSplitter(chunk_size=700, overlap=80)
  separators: "\n\n" → "\n" → ". " → "; " → ", " → " " → ""
  section heading prepended to each chunk's text
                              │
                              ▼
  DocumentChunk(chunk_id="<doc_id>#<sha1[:10]>", doc_id, content,
                source, doc_type, page, section, chunk_index)
  → 49 section-documents → 49 chunks from 8 documents
                              │
                              │ RetrieverManager.ensure_index()
                              ▼
  IndexCache key = "<slug(model)>__<corpus fingerprint>"
  e.g. baai_bge_small_en_v1_5__af02927a088d0c3a
                              │
        cache hit? ───────────┴──► load index.faiss + index.pkl + chunks.json (no embedding work)
             │ no
             ▼
  HuggingFaceEmbeddings.embed_documents(contents) → (49, 384) float32
             │
             ▼
  FAISS.from_embeddings(..., distance_strategy=MAX_INNER_PRODUCT)
  save_local(directory) + chunks.json
```

Why a fingerprint: corpus files (name, size, mtime) plus the chunking config are hashed into the
cache key. Change the corpus, the chunker or the embedding model and a new directory is used, so a
cached index can never serve stale chunks.

---

## 6. Workflow B — answering one question

`RAGPipeline.answer()` builds the prompt and model once in `__init__`:

```python
self.chain = get_prompt(prompt_version) | get_chat_model(settings.generation)
```

so `answer()` only performs retrieval, then one `chain.invoke`.

```
answer(question, question_id)                              pipeline.py
   │ t0 = perf_counter
   ├── retriever.retrieve(question)          exceptions → RAGResponse(error="retrieval failed: …")
   │      │ RetrieverManager.retrieve                        retriever.py
   │      │
   │      │ 1. ensure_index()                 cache load or build
   │      │ 2. strategy = vector | bm25 | hybrid
   │      │
   │      │    vector: embed_query → FAISSVectorStore.search(qv, candidate_k=20)
   │      │    bm25:   BM25Retriever.invoke(q) + vectorizer.get_scores(q)
   │      │    hybrid: EnsembleRetriever.invoke(q)
   │      │              weights={vector:0.6, bm25:0.4}, c=rrf_k (60)
   │      │            weighted alternative: weighted_score_fusion()
   │      │
   │      │ 3. similarity_threshold filter (vector only)
   │      │ 4. candidates = top candidate_k (20)
   │      │ 5. if reranker.enabled:
   │      │       CrossEncoderReranker.compress_documents(candidates, question)
   │      │       → top reranker.top_n (5), timings["rerank"] recorded
   │      │    else: candidates[:top_k]
   │      │ 6. wrap each in RetrievedChunk(rank, score, fused_score, rerank_score)
   │      └── return list[RetrievedChunk]
   │
   ├── context_chunks = retrieved[:generation.context_chunks]
   ├── chain.invoke({"context": format_context(chunks), "question": question})
   │      ChatPromptTemplate → system + human messages (v1 or v2)
   │      ChatHuggingFace (Qwen 0.5B) or ChatOpenAI, temperature=0, max_new_tokens=220
   │      → AIMessage(content)
   ├── any exception → error="generation failed: …"      never a fabricated answer
   │
   └── RAGResponse(answer, retrieved, contexts, sources, latency_ms,
                   retrieval_latency_ms, rerank_latency_ms,
                   generation_latency_ms, error)
```

Notes that matter for reading results:

- `RetrievedChunk.score` is the cross-encoder score when one exists; otherwise the fused/vector/BM25
  score. LangChain's `CrossEncoderReranker` returns documents in rescored order but does not attach
  the cross-encoder value, so `rerank_score` is `None` with that compressor and the fused score is
  what gets reported.
- For `hybrid` + `rrf`, `score` is the RRF weight of the rank the document landed at,
  `1 / (c + rank)`, because `EnsembleRetriever` exposes order rather than fused values.
- Retrieval failures produce an empty-answer response with an `error` string; they are never scored
  as if the model had answered.

---

## 7. Workflow C — scoring one question

`evaluate_item()` in `run_evaluation.py` runs the pipeline once and then computes four groups of
numbers, in this order.

### 7.1 Retrieval metrics — `retrieval_metrics.py`

Relevance is **document-level**: a retrieved chunk is relevant when its `doc_id` is listed in the
question's `expected_context`. Retrieved chunks are de-duplicated by `doc_id` and cut to `top_k`.

| Metric | Definition |
| --- | --- |
| Hit rate | 1 if at least one expected document was retrieved |
| Recall@n | expected documents found in the first n, divided by expected count |
| Precision@K | relevant retrieved / retrieved (after de-duplication) |
| MRR | 1 / rank of the first relevant document |
| Context precision | RAGAS-style rank-weighted precision over the relevance pattern |
| Context recall | relevant retrieved / expected total |

Questions with an empty `expected_context` (unanswerable) are marked `evaluable=False` and excluded
from retrieval aggregates entirely — they are not scored as misses.

### 7.2 Generation metrics — `text_metrics.py`

Deterministic proxies, labelled as such everywhere: no LLM entailment, no judge.

| Metric | Definition |
| --- | --- |
| Faithfulness | share of answer sentences whose best cosine similarity to any context sentence is ≥ 0.55 |
| Answer relevancy | cosine(question, answer) |
| Correctness | `0.7 · cosine(answer, reference) + 0.3 · token F1`, clipped to 0–1 |
| Token F1 | SQuAD-style bag-of-tokens F1 after lowercasing and article/punctuation removal |
| Refusal detection | narrow regex list; "add-ons are not included" is not a refusal |

Unanswerable questions use an explicit rule instead of similarities: a refusal is correct,
an answer is incorrect.

### 7.3 LLM judge — `geval.py` (optional per run)

Five criteria on a 1–5 rubric (`correctness`, `completeness`, `groundedness`, `clarity`,
`conciseness`), normalised as `(score − 1) / 4`. Parsing is line-oriented and tolerant; a criterion
whose line cannot be parsed keeps `None` and the raw text is stored. Judge coverage is reported, so
gaps are visible instead of being filled.

DeepEval (`deepeval_adapter.py`) is a separate, opt-in layer. With `evaluation.deepeval.enabled:
false` — the default — no DeepEval score is produced and the report says so.

### 7.4 Weighted score, thresholds and error class — `scoring.py`, `error_analysis.py`

```
overall = Σ(component · weight) / Σ(weights of available components)
weights: faithfulness .30, answer_relevancy .20, context_recall .20,
         context_precision .15, correctness .15
```

An unavailable component is dropped and the remaining weights renormalised; if nothing is
available the score is `None`. Passing requires the overall score, faithfulness, relevancy,
correctness (and context recall, when the question is answerable) to clear their thresholds in
`configs/evaluation.yaml`.

Error classification is deterministic and ordered; the first matching rule wins:

1. run error → `retrieval_failure` / `generation_failure`
2. answerable but context recall 0 → `retrieval_failure`
3. unanswerable, answered anyway → `hallucination`
4. answerable, refused → `insufficient_context`
5. faithfulness below threshold → `hallucination`
6. score below pass threshold → `generation_failure`
7. no deterministic generation score → `evaluation_failure`
8. otherwise → `none`

A correct refusal on an unanswerable question is reset to `none`, because refusing is the right
answer there.

---

## 8. Workflow D — one experiment run

`run_experiment()` is the orchestrator (CLI and API share it):

```
load_settings(config_path)                       configs merged, validated, tracing configured
   ├── load_dataset(dataset_path)                 validate ids, categories, fields
   ├── limit (optional)                           first N questions only
   ├── load_documents(documents_dir)              corpus ids for expected_context validation
   ├── init_engine(db) + upsert_experiment        status="running", config snapshot stored
   ├── RAGPipeline(settings) + ensure_index()
   ├── embed = get_embedding_model(settings.embedding)
   ├── judge = GevalJudge(settings.evaluation.judge)
   └── for each item: evaluate_item(...)          one answer, one QuestionResult
          │
          ▼
   aggregate(results) → AggregateMetrics
          │
          ├── reports/experiments/<ID>_results.json     per-question records
          ├── ExperimentSummary built (config snapshot, notes, timestamps)
          ├── reports/experiments/<ID>_metrics.txt      numbers only   ← always written
          ├── reports/experiments/<ID>_report.txt       full report    ← unless --no-report
          ├── reports/experiments/<ID>_summary.json
          └── DB: results rows + finish_experiment + report registration
```

Notes recorded in the summary explain what the run did *not* do: judge skipped, DeepEval disabled,
dataset warnings. Those notes appear in the report's RUN NOTES section.

---

## 9. Workflow E — report, metrics file and baseline

Both files are produced from the same `ExperimentSummary`; neither contains guessed values.

**`<ID>_report.txt`** — `build_text_report()`: evaluation date, experiment, dataset, question
count, configuration (retrieval, embedding model, LLM, reranker, prompt version, chunking), the
metric table, observations derived only from measured values, recommendations tied to the signals
that triggered them, run notes.

**`<ID>_metrics.txt`** — `build_metrics_report()`: the same numbers with no prose, in sections

```
RETRIEVAL  hit rate, recall@1/3/5/10, precision@K, MRR, contextual precision/recall
GENERATION faithfulness, answer relevancy, correctness, token F1
LLM JUDGE  G-Eval criteria + judge coverage          (only when computed)
DEEPEVAL   the four scores                            (only when enabled)
OVERALL    overall score, threshold pass rate, hallucination rate, answered
LATENCY    mean / p50 / p95 / p99, retrieval / rerank / generation means
FAILURE CLASSIFICATION   per-category counts
```

**Baseline** — `reports/baseline/baseline.txt` + `baseline.json`, written only by an explicit
`--save-baseline` or `POST /api/baseline`. Nothing in the evaluation path reads or writes it, so a
run can never overwrite the stored baseline by accident.

---

## 10. Workflow F — API request lifecycle

```
POST /api/evaluations  {experiment_id, config_path, dataset_path?, limit?, skip_judge?, generate_report?}
   ├── validate config file exists                 → 404 otherwise
   ├── JOBS[id] = {status: "queued"}               in-process job registry
   ├── BackgroundTasks.add_task(_run_job, ...)     → 202 immediately
   └── _run_job: run_experiment(...)               status running → completed | failed
```

| Method | Path | Handler |
| --- | --- | --- |
| GET | `/api/health` | `health` |
| GET | `/` | `index` (HTML: baseline text + stored runs) |
| GET | `/api/baseline` | `get_baseline` |
| POST | `/api/baseline` | `update_baseline` (explicit replacement) |
| POST | `/api/datasets` | `upload_dataset` |
| GET | `/api/datasets` | `list_datasets` |
| POST | `/api/experiments` | `create_experiment` (full dataset) |
| POST | `/api/evaluations` | `create_evaluation` (optional `limit`) |
| GET | `/api/experiments` | `list_experiments` |
| GET | `/api/experiments/{id}` | `get_experiment` |
| GET | `/api/evaluations/{id}` | `get_evaluation` |
| GET | `/api/evaluations/{id}/results` | `get_evaluation_results` |
| GET | `/api/reports/{id}` | `get_report` (regenerates the txt if missing) |
| GET | `/api/metrics/{id}` | `get_metrics` (regenerates the metrics file if missing) |

Jobs live in process memory, so a server restart forgets queued/running status. Completed runs are
in SQLite and survive restarts.

---

## 11. Configuration

Resolution order (later wins):

```
configs/config.yaml → configs/models.yaml → configs/evaluation.yaml
                    → --config configs/experiments/<experiment>.yaml
                    → .env (KEY=VALUE, never committed)
```

Everything is validated by Pydantic at load time, including cross-field rules:

- `MetricWeights` must sum to 1.0 and be non-negative.
- `RetrievalConfig` requires `candidate_k ≥ top_k`.
- `HybridConfig` rejects all-zero fusion weights.
- `ChunkingConfig` requires `0 ≤ chunk_overlap < chunk_size`.
- `ExperimentMeta.experiment_id` must start with `EXP-`.
- enabling `hybrid` turns on BM25 and switches `retrieval.strategy` to `hybrid`; asking for
  `bm25`/`hybrid` always enables BM25.

Section reference:

| Section | Controls |
| --- | --- |
| `paths` | documents, datasets, cache, reports, database, logs |
| `embedding` | model, device, batch size, normalisation, BGE query instruction |
| `chunking` | strategy (`recursive` / `fixed`), chunk size, overlap |
| `retrieval` | strategy, top_k, candidate_k, similarity_threshold |
| `bm25` | enabled, k1, b |
| `hybrid` | enabled, fusion (`rrf` / `weighted`), rrf_k, weights |
| `reranker` | enabled, model, top_n |
| `generation` | provider, model, temperature, max_new_tokens, context_chunks, prompt_version |
| `evaluation` | metric weights, thresholds, judge, DeepEval, reference similarity |
| `tracing` | enabled, project, endpoint, api_key_env — key always from the environment |
| `api` | host, port |

### LangSmith tracing

LangChain calls can be traced without code changes. `tracing.enabled: true` in
`configs/config.yaml` plus `LANGCHAIN_API_KEY` in the environment (or whatever
`tracing.api_key_env` names) sets `LANGCHAIN_TRACING_V2`, `LANGCHAIN_PROJECT` and
`LANGCHAIN_ENDPOINT`. If the flag is on but the key is missing, a warning is logged and the run
continues untraced — evaluations never fail because tracing is misconfigured.

---

## 12. Prompt v1 vs v2 — `prompts.py`

Both are `ChatPromptTemplate`s with a system message and a human message that receives
`{context}` and `{question}`. The only difference is the instruction block.

| | v1 (baseline) | v2 |
| --- | --- | --- |
| System | "helpful assistant" | "answer strictly from the supplied documents, never invent facts" |
| Grounding rules | none | sentence-level source citations, no outside knowledge |
| Unanswerable | not addressed | exact refusal sentence |
| Purpose | measure the naive prompt | measure the effect of explicit grounding |

Context is rendered as `[doc_id page N]` blocks separated by `---`, so the model sees a citable
source id per chunk and the refusal sentence can be checked mechanically.

---

## 13. Honesty rules encoded in the code

- Missing metric → `None` → renormalised weights or `N/A`. Never `0`, never a default.
- Judge output that cannot be parsed → `None` score plus the raw text, and coverage is reported.
- DeepEval off → the report states that no DeepEval scores exist.
- Retrieval/generation failure → recorded on the question, excluded from generation averages.
- Unanswerable questions are excluded from retrieval metrics and scored by an explicit rule.
- Every metric is labelled by how it was produced: deterministic proxy, LLM-as-a-judge, or DeepEval.
- The numerical knobs behind those rules are module constants with names that say what they are:
  `SUPPORT_SIMILARITY = 0.55`, `REFUSAL_PATTERNS`, `WEIGHT_KEYS`, and the threshold block in
  `configs/evaluation.yaml`.

---

## 14. Tests

`pytest tests -q` → 40 tests, no model downloads, no network.

| File | Covers |
| --- | --- |
| `test_config.py` | weight sum validation, hybrid flags, `candidate_k ≥ top_k` |
| `test_dataset.py` | dataset shape, unknown `expected_context`, duplicate/missing fields |
| `test_chunking.py` | markdown sections become documents, section metadata, chunk size, corpus loads |
| `test_fusion.py` | ensemble prefers agreement, weights respected, weighted normalisation |
| `test_prompts.py` | v1 has no refusal rule, v2 does, unknown version rejected |
| `test_text_metrics.py` | token F1, refusal detection, unanswerable rule |
| `test_retrieval_metrics.py` | perfect / miss / partial / dedup / unanswerable |
| `test_scoring.py` | weights, renormalisation, thresholds, hallucination rules |
| `test_text_report.py` | report sections and values, baseline persistence, metrics file |

---

## 15. Stored results

Measured on the earlier 25-question corpus (before two binary documents were removed). The stored
reports were **not** rewritten, so they describe that run; a fresh run on the current 8-document,
23-question corpus will produce new numbers.

| Metric | EXP-001 (vector, v1) | EXP-002 (hybrid, v2, reranker) |
| --- | ---: | ---: |
| Overall weighted score | 0.902 | 0.939 |
| Correctness | 0.751 | 0.810 |
| Token F1 | 0.545 | 0.602 |
| Answer relevancy | 0.854 | 0.824 |
| Precision@K | 0.454 | 0.424 |
| Hit rate / Recall@1 / MRR | 1.000 / 0.957 / 1.000 | unchanged |
| Pass rate | 92.0% | 92.0% |
| Hallucination rate | 4.0% | 0.0% |
| Mean latency | 13.18 s | 8.25 s |

Reading: the gain is generation-side (prompt v2 grounded the answers and produced one refusal
where the baseline hallucinated). Retrieval was already at ceiling on a small corpus, so hybrid
retrieval and reranking cost time without improving recall. The bundle is not presented as an
ablation and is not claimed to generalise.

---

## 16. Extension points

| Want to… | Change |
| --- | --- |
| add a model | add it to `configs/models.yaml`, point `embedding.model` / `generation.model` at it |
| add a prompt version | add an entry to `PROMPT_TEXT` in `prompts.py` |
| add a retrieval strategy | extend `RetrieverManager.retrieve` with a new branch |
| add a metric | add the field to `GenerationMetrics`/`RetrievalMetrics`, compute it in the metric module, render it in `build_metrics_report` and the report |
| change weighting | edit `configs/evaluation.yaml` (validated at load) |
| run a variant | copy an experiment YAML and change one setting |
| use a hosted judge or LLM | set the provider to `openai_compatible` and export `OPENAI_API_KEY` |
| turn on DeepEval | `evaluation.deepeval.enabled: true` |
| trace runs | `tracing.enabled: true` + `LANGCHAIN_API_KEY` |

---

## 17. Known limitations

- 23 questions is a small sample; a single question moves an aggregate by several points.
- Reference answers are not exhaustive, so correctness under-rewards valid paraphrases.
- The default generator and judge are small CPU models; scores are directional, not authoritative.
- Faithfulness is a cosine-similarity proxy, not entailment.
- The reranker's raw scores are not exposed by LangChain's compressor, so `rerank_score` stays
  `None` and the fused score is reported instead.
- Retrieval relevance is graded per document, not per chunk.
