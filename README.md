# News Briefing Agent

A news briefing agent that takes a topic, searches the web for recent articles, evaluates them, and produces a structured markdown briefing with inline citations. Built in two parallel architectures — a linear single-agent pipeline and a supervisor-style multi-agent pipeline — specifically to measure where multi-agent coordination actually earns its cost.

The interesting work isn't the briefings themselves; it's the evaluation infrastructure around them. Source-tier classifier, two-part faithfulness judge (Haiku for per-claim support, Sonnet for retrieval utilization), cross-pipeline retrieval-overlap invariant. Per-cell artifacts on disk, aggregation into CSV + markdown comparison tables, caveats surfaced at read-time rather than hidden in footnotes.

## Architecture

Two pipelines share a state schema and the same Researcher sub-agent. Everything else diverges.

```
Single-agent (src/graph.py) — Quadrant 2: linear workflow + one embedded agent
  planner  →  researcher  →  filter  →  summarizer  →  formatter

Multi-agent (src/graph_multi.py) — Quadrant 4: workflow with multiple agents
  planner  →  researcher  →  critic_articles  →  writer  →  critic_draft
                                                             ↑         │
                                                             └ revise ─┘
                                                    (conditional edge,
                                                     revision cap = 2)
```

The supervisor in the multi-agent pipeline is a Python conditional edge, not an LLM node. The only genuinely model-driven control-flow decision is `critic_draft`'s accept/revise verdict. Everything else is deterministic routing.

See `DESIGN.md` for the full architectural story, decision log, and tradeoff analysis.

## Quick start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
export TAVILY_API_KEY=...

# Single-agent briefing (CLI)
python -m src.cli "EU AI Act enforcement across member states"

# Multi-agent briefing (Python)
python -c "import asyncio; from src.runner import run_briefing_multi_agent; \
           r = asyncio.run(run_briefing_multi_agent('EU AI Act enforcement')); \
           print(r.briefing_markdown if r.status == 'success' else r.message)"
```

## Repo map

```
src/
  graph.py            single-agent pipeline (linear)
  graph_multi.py      multi-agent pipeline (supervisor)
  runner.py           error-handling boundary; BriefingResult contract
  schemas.py          Pydantic models + TypedDict state classes
  nodes/              node wrappers (planner, researcher, filter, ...)
  agents/             LLM-backed agent bodies (critic_articles, writer, ...)
  tools/
    web_search.py     Tavily wrapper with tenacity retries
    tavily_cache.py   record/replay cache for reproducible retrieval

evals/
  eval_set.json       5-topic golden set with per-topic expectations
  run_evals.py        harness: runs every (topic × pipeline) cell
  source_tiers.py     deterministic source-tier classifier (no LLM)
  llm_judge.py        two-part LLM-as-judge (Haiku P1, Sonnet P2)
  retrieval_overlap.py per-topic Jaccard invariant across pipelines
  aggregate.py        reads all metric JSONs, writes comparison.csv + .md
  results/current/    per-cell artifacts + aggregated comparison

tests/                37 tests across schemas, graph topology, metrics

DESIGN.md             architecture, decisions, tradeoffs
SPEC.md               contracts + implementation details
FAILURES.md           observed failure modes with root cause and mitigation
notes/                running learning journal (day1_notes → day7_summary)
```

## Evaluation

The harness produces per-cell artifacts; metrics are separate passes over the artifacts. This keeps metric re-runs free once the briefings are generated, and lets each metric module fail independently.

```bash
# Produce per-cell briefing artifacts (only rerun when pipeline changes)
PYTHONPATH=. python evals/run_evals.py

# Run each metric pass (free to rerun after prompt tweaks; disk-cached)
PYTHONPATH=. python evals/source_tiers.py --write-scores
PYTHONPATH=. python evals/retrieval_overlap.py --write
PYTHONPATH=. python evals/llm_judge.py --write-scores

# Aggregate all metric JSONs into comparison.csv + comparison.md
PYTHONPATH=. python evals/aggregate.py
```

The three metrics measure different failure modes and are reported separately — no composite score. Collapsing source-tier, claim support, and retrieval utilization into one number would let a pipeline improve on one while degrading on another, and the score would hide that.

### Headline results (5-topic run)

| | single-agent | multi-agent |
|---|---|---|
| Median briefing length (chars) | 9,714 | 4,834 |
| Median citation count | 41 | 17 |
| UGC citations appear in briefing | 2 of 5 topics | 0 of 5 topics |
| Per-claim support rate (Haiku judge) | 69–94% | 58–100% |
| Retrieval utilization (Sonnet judge) | 0.60–1.00 | 0.60 on all 5 |

The cleanest cross-pipeline story: multi-agent trims length, drops UGC sources, and tilts toward higher-tier citations — but on EU AI Act specifically, the Critic dropped civil-society coverage that Part 2 of the judge flagged as substantive omission. Every architecture choice has a failure mode; the eval surfaces which.

See `evals/results/current/comparison.md` for the full per-topic breakdown with retrieval-pinning caveats inline.

## Observability

Every run emits a LangSmith trace with a `thread_id` printed on completion. Traces surface node-level spans, tool-call arguments, token counts, and duration per span. Used extensively during development for debugging (5 documented failure modes in `FAILURES.md`, each with a specific trace anchor).

## Known limitations

The project is a learning artifact, not a production system. Honest limits as of the end of the build:

- **Cross-pipeline retrieval pinning is incomplete.** The Tavily cache keys on query string, not on topic. The Researcher sub-agent rephrases queries based on upstream state, so on 2 of 5 topics the two pipelines saw meaningfully different article sets (Jaccard 0.47 and 0.54). `retrieval_overlap.py` detects this; mitigation is pending. See FAILURES.md entry.
- **Part 2 judge resolution is coarse.** With a cap of 5 high-importance omissions, the metric takes three values on this run (0.60, 0.80, 1.00). Discriminates at the extremes; less useful for comparing close cells.
- **No live human-in-the-loop.** The Writer → Critic(draft) → conditional-edge shape is structurally isomorphic to HITL (would be a one-node swap to LangGraph's `interrupt` primitive), but it's not wired.
- **Read-only pipeline.** No database writes, no external action-taking. The prompt-injection hardening on `critic_articles` is a transferable pattern, not a battle-tested production defense.
- **Single-family judge.** Claude judges Claude. Cross-family validation (e.g., GPT judging Claude) is the obvious next validation step and is not done.

## Documentation

- **`DESIGN.md`** — architecture walkthrough, decision log per design choice, tradeoff analysis per pipeline
- **`SPEC.md`** — implementation contracts (state schema, node signatures, failure taxonomy)
- **`FAILURES.md`** — observed failure modes, each with root cause, mitigation status, and interview relevance
- **`notes/`** — daily build journal from day 1 through day 7
- **`evals/results/current/comparison.md`** — metric comparison table with caveats inline

## Testing

```bash
pytest
```

37 tests across schema validation, graph topology, runner input validation, Tavily cache record/replay, source-tier classification, LLM judge (mocked), and retrieval overlap.