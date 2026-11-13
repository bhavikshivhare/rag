# RAG-Bench — Run Commands

Every command runs from the project root (the folder containing this file) using the project's local
interpreter `.venv/Scripts/python.exe`. Paths are written Git-Bash style; in PowerShell use `\`
instead of `/`, and `$env:VAR="value"` instead of `export VAR=value`.

Shortcut wrappers exist for the common flows: `./run.sh <target>` (Git Bash / WSL) and
`.\run.ps1 <target>` (PowerShell). Targets: **setup, ingest, baseline, enhanced, reports, api,
test, all**.

> On a non-UTF-8 Windows console, prefix a command with `PYTHONIOENCODING=utf-8` to avoid
> `UnicodeEncodeError` on `→` and similar characters.

---

## 0. One-time setup

```bash
# virtualenv (built and verified on Python 3.10.5; 3.11+ works)
py -3.10 -m venv .venv

# CPU-only torch first (much smaller than the CUDA default build)
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1

# the rest of the stack (LangChain, FAISS, FastAPI, DeepEval, ...)
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Or simply:

```bash
./run.sh setup
```

The default configuration is fully offline: embeddings `BAAI/bge-small-en-v1.5`, generator and judge
`Qwen/Qwen2.5-0.5B-Instruct`, reranker `BAAI/bge-reranker-base`. The first run downloads roughly
1.2 GB into `.cache/huggingface`.

Optional — only for an external OpenAI-compatible LLM/judge:

```bash
cp .env.example .env      # then set OPENAI_API_KEY / OPENAI_BASE_URL
```

### Optional: LangSmith tracing

```bash
export LANGCHAIN_API_KEY=...      # PowerShell: $env:LANGCHAIN_API_KEY="..."
```

Then set `tracing.enabled: true` in `configs/config.yaml` (and optionally `tracing.project`). Every
LangChain call — retrieval, fusion, reranking, prompt, model — is then traced. With the flag on but
no key, a warning is logged and the run continues untraced.

---

## 1. Corpus and dataset

```bash
# what is in the corpus (8 Markdown documents)
ls data/documents

# validate the benchmark: ids, required fields, categories, expected_context
.venv/Scripts/python.exe -m src.evaluation.dataset data/datasets/benchmark_v1.jsonl
```

The dataset is 23 questions over 8 documents: `factual`, `comparison`, `multi-hop`,
`summarization` and two `unanswerable`. `expected_context` holds document ids; ids that do not exist
in the corpus are reported as warnings at run time.

---

## 2. Ingest and build the index

```bash
# statistics only - per-document chunk counts, no index build
.venv/Scripts/python.exe -m src.rag.ingestion --stats

# ingest + build (or reuse) the FAISS index cache
.venv/Scripts/python.exe -m src.rag.ingestion
```

The index is cached in `data/cache/index/<embedding-model>__<corpus-fingerprint>/` — for the current
corpus: 49 chunks from 8 documents. Re-running with the same corpus, chunking and embedding model
reuses it; changing any of them creates a new directory. Files written: `index.faiss`, `index.pkl`,
`chunks.json`.

Wrapper: `./run.sh ingest`.

---

## 3. Run an experiment

```bash
# EXP-001 - vector retrieval, prompt v1, no reranker
.venv/Scripts/python.exe -m src.evaluation.run_evaluation \
  --config configs/experiments/exp_001_baseline.yaml

# EXP-002 - hybrid BM25+vector (RRF), prompt v2, cross-encoder reranker
.venv/Scripts/python.exe -m src.evaluation.run_evaluation \
  --config configs/experiments/exp_002_enhanced.yaml
```

Wrappers: `./run.sh baseline` and `./run.sh enhanced`.

### Flags

| Flag | Effect |
| --- | --- |
| `--config PATH` | experiment YAML, merged over `configs/config.yaml` (required) |
| `--dataset PATH` | run a different benchmark JSONL |
| `--limit N` | evaluate only the first N questions (smoke runs) |
| `--skip-judge` | skip the G-Eval judge generation (faster, judge scores absent) |
| `--no-report` | skip the full text report (the metrics file is still written) |

Quick smoke run (~2 minutes):

```bash
.venv/Scripts/python.exe -m src.evaluation.run_evaluation \
  --config configs/experiments/exp_002_enhanced.yaml --limit 3 --skip-judge --no-report
```

⚠️ A run writes into **its experiment id** from the YAML. Re-running `exp_001_baseline.yaml` replaces
`EXP-001`'s stored results, report and database rows. To keep existing runs, copy the YAML and change
`experiment.experiment_id` (for example `EXP-003`).

---

## 4. What a run produces

| Output | Contents |
| --- | --- |
| `reports/experiments/EXP-00N_report.txt` | full report: config, metrics, observations, recommendations, run notes |
| `reports/experiments/EXP-00N_metrics.txt` | the same numbers only, grouped by section |
| `reports/experiments/EXP-00N_results.json` | one record per question |
| `reports/experiments/EXP-00N_summary.json` | aggregate metrics + config snapshot + timestamps |
| `data/ragbench.db` | SQLite: datasets, experiments, per-question results, report paths |
| `logs/*.log` | loguru log for the run |

No PDF, DOCX, Excel or HTML file is produced.

---

## 5. Report, metrics file and baseline

```bash
# re-render the full text report from stored results (no re-evaluation)
.venv/Scripts/python.exe -m src.evaluation.text_report --experiment-id EXP-001

# write only the metrics file
.venv/Scripts/python.exe -m src.evaluation.text_report --experiment-id EXP-002 --metrics

# store a run as the baseline (explicit request only)
.venv/Scripts/python.exe -m src.evaluation.text_report --experiment-id EXP-001 --save-baseline

# print the saved baseline
.venv/Scripts/python.exe -m src.evaluation.text_report --show-baseline
```

The baseline lives in `reports/baseline/baseline.txt` (+ `baseline.json`). Evaluation runs never read
or write it; only `--save-baseline` or `POST /api/baseline` replaces it.

Wrapper: `./run.sh reports` (re-renders the EXP-001 report).

---

## 6. API server

```bash
.venv/Scripts/python.exe -m src.api.main
# or
.venv/Scripts/python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Docs at <http://127.0.0.1:8000/docs>. Wrapper: `./run.sh api`.

```bash
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/experiments

# start a run in the background (202 immediately)
curl -X POST http://127.0.0.1:8000/api/evaluations \
  -H "Content-Type: application/json" \
  -d '{"experiment_id":"EXP-003","config_path":"configs/experiments/exp_002_enhanced.yaml","limit":5}'

# poll status, then per-question results
curl http://127.0.0.1:8000/api/evaluations/EXP-003
curl http://127.0.0.1:8000/api/evaluations/EXP-003/results

# plain-text outputs
curl http://127.0.0.1:8000/api/reports/EXP-003     # full report
curl http://127.0.0.1:8000/api/metrics/EXP-003     # metrics only
curl http://127.0.0.1:8000/api/baseline            # saved baseline

# explicitly store an experiment as the baseline
curl -X POST http://127.0.0.1:8000/api/baseline \
  -H "Content-Type: application/json" -d '{"experiment_id":"EXP-003"}'

# upload another benchmark
curl -X POST http://127.0.0.1:8000/api/datasets -F "file=@data/datasets/benchmark_v1.jsonl"

# minimal HTML view (baseline text + stored runs)
start http://127.0.0.1:8000/
```

Job status (`queued` / `running` / `completed` / `failed`) lives in process memory: a restart forgets
in-flight jobs, but completed runs are in SQLite.

---

## 7. Tests

```bash
.venv/Scripts/python.exe -m pytest tests -q
```

40 unit tests, no model downloads. Wrapper: `./run.sh test`.

---

## 8. Wrapper targets

```bash
./run.sh setup       # pip install -r requirements.txt
./run.sh ingest      # ingest + build/reuse the index
./run.sh baseline    # EXP-001
./run.sh enhanced    # EXP-002
./run.sh reports     # re-render the EXP-001 text report
./run.sh api         # uvicorn on 127.0.0.1:8000
./run.sh test        # pytest
./run.sh all         # dataset validation + ingest + EXP-001 + EXP-002
```

`.\run.ps1 <target>` accepts the same set. `all` re-runs both experiments under their own ids — see
the warning in section 3.

---

## 9. Docker

```bash
docker build -t rag-bench .
docker run --rm \
  -v "%cd%/data:/app/data" \
  -v "%cd%/reports:/app/reports" \
  -v "%cd%/.cache:/app/.cache" \
  rag-bench
```

The image installs `requirements.txt` and its default command runs the baseline experiment; mount
`data/`, `reports/` and `.cache/` to keep the corpus, database, outputs and downloaded model weights
on the host instead of inside the container.

---

## 10. Reset caches

```bash
rm -rf data/cache/index        # force an index rebuild on the next run
rm -f data/ragbench.db         # drop stored results (saved .txt files stay on disk)
rm -rf reports/baseline        # forget the stored baseline
rm -rf .cache/huggingface      # delete ~1.2 GB of downloaded model weights
```

---

## 11. Measured on this machine

CPU-only, Windows, 16 GB RAM, models already cached.

| Step | Time |
| --- | --- |
| ingestion + index build | < 1 min |
| EXP-001, 25 questions with G-Eval judge | ~23 min |
| EXP-002, 25 questions with reranker + G-Eval judge | ~9 min |
| metrics file / report re-render | instant |

Stored results (25-question set, before two binary documents were removed — the stored reports were
not rewritten):

| Metric | EXP-001 | EXP-002 |
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

The improvement is generation-side (prompt v2). Retrieval was already at ceiling on this corpus, so
hybrid + reranking cost time without improving recall. Neither report claims the bundle generalises.

---

## 12. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `FileNotFoundError: documents directory not found` | run from the project root, or fix `paths.documents` |
| `no supported documents in …` | corpus files must be `.md`, `.txt`, `.pdf` or `.docx` |
| `[WinError 32] … being used by another process` | a corpus file is open in Word/Excel; close it and re-run ingestion |
| First run takes minutes before any output | model download (embeddings, reranker, Qwen) into `.cache/huggingface` |
| Index rebuilt every run | cache key covers corpus mtime, chunking and embedding model — any change invalidates it |
| `UnicodeEncodeError` in the console | prefix with `PYTHONIOENCODING=utf-8` |
| Judge scores missing | `--skip-judge` was used, `evaluation.judge.enabled: false`, or the judge output could not be parsed (see `run notes`) |
| `OPENAI_API_KEY is not set` | provider is `openai_compatible` but no key in `.env` / environment |
| API returns 404 for a report | the experiment id is not in `data/ragbench.db`; list `/api/experiments` |
| LangSmith shows nothing | `tracing.enabled: true` and `LANGCHAIN_API_KEY` must both be set before the run starts |
