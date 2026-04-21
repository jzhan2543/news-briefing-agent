# Evaluation Harness

Infrastructure for running the news-briefing pipelines against a pinned
golden eval set, writing raw run artifacts that downstream metric passes
consume. The harness produces evidence; scoring the evidence is a
separate concern run over the results directory.

## Layout

```
evals/
├── eval_set.json             # 5 topics with trajectory + source expectations
├── run_evals.py              # Harness: topic × pipeline → run artifacts
├── cache/
│   └── tavily/               # Record/replay for Tavily responses (gitignored)
│       └── <query_hash>.json
└── results/
    └── <run_id>/
        └── <topic_id>/
            └── <pipeline>/   # single_agent | multi_agent
                ├── briefing.md
                └── metadata.json
```

`run_id` defaults to `"current"` so re-runs accumulate in one place.
Pass `--run-id <name>` when you want to snapshot results — e.g., before
changing a prompt and re-running, create a branch named for what
you're about to change.

## Why retrieval is pinned

The Day 5 PM comparison confounded architecture with retrieval variance:
Tavily returned different article sets across two runs of the same
query, and the single-agent-vs-multi-agent comparison on that topic
turned into a retrieval-luck comparison. See DESIGN.md §5 (central bank
case) and §7 item 1.

The cache layer (`src/tools/tavily_cache.py`) is the fix. When
`TAVILY_CACHE_DIR` is set, every Tavily call is keyed on its normalized
query, written to disk on first encounter, and replayed from disk on
subsequent calls. The harness sets this up automatically. Production
invocations (anywhere `TAVILY_CACHE_DIR` is unset) are unaffected.

## Usage

First run populates the cache. Subsequent runs of the same topics
replay from disk — no Tavily calls, deterministic article sets, and
an architectural comparison that means what you want it to mean.

```bash
# Run every topic × pipeline in the eval set. Populates the cache on
# first invocation; skip-if-exists on cells with existing metadata.json.
PYTHONPATH=. python evals/run_evals.py

# Narrow to specific topics (by id in eval_set.json)
PYTHONPATH=. python evals/run_evals.py --topics tariffs_2026 uk_budget_2026

# Narrow to one pipeline
PYTHONPATH=. python evals/run_evals.py --pipelines single_agent

# Fresh run under a new run_id so the results tree is snapshotted
PYTHONPATH=. python evals/run_evals.py --run-id day6_morning_baseline

# Re-run every cell regardless of existing output
PYTHONPATH=. python evals/run_evals.py --force

# Blow away the article cache and re-fetch from Tavily on this run.
# Use sparingly — invalidates any prior run's architectural comparison.
TAVILY_CACHE_REFRESH=1 PYTHONPATH=. python evals/run_evals.py --force
```

## Cells, skip-if-exists, resumability

A "cell" is one `(run_id, topic_id, pipeline)` triple. The harness writes
`briefing.md` and `metadata.json` side by side; a cell is considered
complete when `metadata.json` exists. Interrupting mid-run and restarting
is safe — completed cells are skipped, incomplete ones are re-run.

If you want a fresh run, either change `--run-id` (recommended, snapshots
the prior run) or pass `--force` (overwrites in place).

## What metadata.json contains

Enough to run the three planned metrics without re-invoking the pipeline:

- `source_urls` — deterministic source-quality check against a
  reputable-domain allowlist.
- `thread_id` — LangSmith trace lookup for trajectory-correctness
  analysis (did the Researcher make the expected number of tool calls?
  Did the Critic-Writer revision loop fire when we expected?).
- `briefing_chars`, `duration_s` — cheap signals (is multi-agent still
  producing tighter briefings? is it paying for that tightness?).
- `failure_reason`, `failure_message` — flaky runs are part of the
  evidence; they shouldn't silently disappear.

Faithfulness (LLM-as-judge on claims against source snippets) reads
`briefing.md` and the raw article set; both are addressable from this
directory.

## What's deliberately not here yet

- No metric computation. `run_evals.py` produces artifacts; a separate
  pass (planned for the afternoon block) reads the results directory
  and writes scores. Separating "produce" from "score" means re-scoring
  is free.
- No LLM-as-judge. Same reason — it's a metric pass over `briefing.md`
  and source snippets, not part of the harness.
- No 10-15 topic set yet. Starting at 5 to validate the harness shape;
  expand once the metric passes exist and the starting 5 have shown
  whether the current setup measures what we want it to.

## Interview anchor

The cache + harness is what promotes the offline-eval story from
theoretical to operational. If asked about how you'd evaluate an
agentic system without ground truth per-run, the answer is: you build
a pinned golden set, you compare architectural variants on byte-identical
inputs, and you separate the runs from the scoring so iteration on
either axis is cheap. This repo does all three.