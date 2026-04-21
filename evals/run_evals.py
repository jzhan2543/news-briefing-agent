"""
Evaluation harness: runs the golden eval set through both pipelines with
Tavily retrieval pinned via the cache layer, writes raw run artifacts to
disk. No metrics computed here — metric passes read from the results
directory.

Design decisions:

1. Sequential execution. Six briefings in parallel would confuse
   rate-limit symptoms with real failure modes, and the wall-clock
   comparison between pipelines needs to be apples-to-apples. Same
   rationale as scratch/run_comparison.py.

2. Cache dir is set by the harness, not inherited from env. The harness
   IS the opt-in path for the cache layer. Callers shouldn't have to
   remember to export TAVILY_CACHE_DIR; running this script pins
   retrieval automatically. If the user wants to refresh the cache, they
   export TAVILY_CACHE_REFRESH=1 before invoking.

3. Skip-if-exists on the (run_id, topic, pipeline) cell. A partial run
   is always resumable. Default run_id is "current" so re-invocations
   accumulate in one place; pass --run-id <name> to branch.

4. Artifacts not metrics. The harness writes briefing markdown + a
   per-run JSON with duration/thread_id/source URLs/failure info.
   Metric computation (source-quality, trajectory, faithfulness) runs
   against this directory later and doesn't re-invoke the pipeline.

Usage:
    PYTHONPATH=. python evals/run_evals.py
    PYTHONPATH=. python evals/run_evals.py --run-id day6_morning
    PYTHONPATH=. python evals/run_evals.py --topics tariffs_2026 uk_budget_2026
    PYTHONPATH=. python evals/run_evals.py --pipelines single_agent
    PYTHONPATH=. python evals/run_evals.py --force
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
EVALS_DIR = REPO_ROOT / "evals"
EVAL_SET_PATH = EVALS_DIR / "eval_set.json"
RESULTS_DIR = EVALS_DIR / "results"
CACHE_DIR = EVALS_DIR / "cache" / "tavily"

PIPELINES = ("single_agent", "multi_agent")


# ---------------------------------------------------------------------------
# Cache activation — set BEFORE importing src.runner (which imports
# src.tools.web_search at module load). The cache layer reads env at
# call time, not import time, so import order doesn't matter for
# correctness — but setting it up here keeps the activation sequence
# explicit and auditable.
# ---------------------------------------------------------------------------

def _activate_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TAVILY_CACHE_DIR"] = str(cache_dir)
    # TAVILY_CACHE_REFRESH is deliberately NOT set here — a caller who
    # wants to refresh the cache exports it themselves before invoking.
    # That keeps the "normal" harness run idempotent.


# ---------------------------------------------------------------------------
# Result artifacts
# ---------------------------------------------------------------------------

@dataclass
class CellArtifact:
    """What we write per (run_id, topic, pipeline) cell.

    Enough to run the source-quality, trajectory, and faithfulness
    metrics later without re-invoking the pipeline. Failure cells are
    also captured so flaky runs are visible in the results tree rather
    than silently missing.
    """

    run_id: str
    topic_id: str
    topic: str
    pipeline: str  # "single_agent" | "multi_agent"
    status: str    # "success" | "failure"

    run_started_at: str   # ISO8601 UTC
    run_completed_at: str # ISO8601 UTC
    duration_s: float

    # Populated on success
    thread_id: str | None = None
    source_urls: list[str] | None = None
    briefing_chars: int | None = None

    # Populated on failure
    failure_reason: str | None = None
    failure_message: str | None = None


def cell_dir(run_id: str, topic_id: str, pipeline: str) -> Path:
    return RESULTS_DIR / run_id / topic_id / pipeline


def briefing_path_for(run_id: str, topic_id: str, pipeline: str) -> Path:
    return cell_dir(run_id, topic_id, pipeline) / "briefing.md"


def metadata_path_for(run_id: str, topic_id: str, pipeline: str) -> Path:
    return cell_dir(run_id, topic_id, pipeline) / "metadata.json"


def write_cell(artifact: CellArtifact, briefing_markdown: str) -> None:
    """Persist one cell's outputs. briefing.md + metadata.json, side by
    side. JSON is the machine-readable version; markdown is what humans
    and downstream LLM-as-judge calls read."""
    out_dir = cell_dir(artifact.run_id, artifact.topic_id, artifact.pipeline)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "briefing.md").write_text(briefing_markdown)
    (out_dir / "metadata.json").write_text(
        json.dumps(asdict(artifact), indent=2, sort_keys=True)
    )


def cell_exists(run_id: str, topic_id: str, pipeline: str) -> bool:
    """A cell is considered complete when metadata.json exists. briefing.md
    might also be there; absence of metadata.json means a prior run was
    interrupted mid-write and the cell should be re-run."""
    return metadata_path_for(run_id, topic_id, pipeline).exists()


# ---------------------------------------------------------------------------
# Eval set loading
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalTopic:
    """Just the fields the harness itself needs. The full eval set JSON
    carries much more (must_cite_domains, trajectory_expectations, ideal
    briefing sketch) — those fields matter to metric passes, not to the
    harness that produces the raw runs."""
    id: str
    topic: str


def load_eval_topics(path: Path, filter_ids: list[str] | None = None) -> list[EvalTopic]:
    data = json.loads(path.read_text())
    topics = [EvalTopic(id=t["id"], topic=t["topic"]) for t in data["topics"]]
    if filter_ids:
        wanted = set(filter_ids)
        topics = [t for t in topics if t.id in wanted]
        missing = wanted - {t.id for t in topics}
        if missing:
            raise SystemExit(
                f"Unknown topic IDs: {sorted(missing)}. "
                f"Available: {[t.id for t in load_eval_topics(path)]}"
            )
    return topics


# ---------------------------------------------------------------------------
# The actual runs
# ---------------------------------------------------------------------------

async def _invoke(pipeline: str, topic: str):
    """Dispatch to the right runner. Lazy-imported so the harness
    activates the cache before the runner's module load chain touches
    the web_search module — not strictly required (cache reads env at
    call time) but keeps the activation order auditable.
    """
    from src.runner import run_briefing, run_briefing_multi_agent

    if pipeline == "single_agent":
        return await run_briefing(topic)
    if pipeline == "multi_agent":
        return await run_briefing_multi_agent(topic)
    raise ValueError(f"unknown pipeline: {pipeline}")


async def run_cell(
    run_id: str,
    topic: EvalTopic,
    pipeline: str,
    force: bool,
    cell_number: int,
    total_cells: int,
) -> None:
    from src.schemas import BriefingFailure, BriefingSuccess

    if cell_exists(run_id, topic.id, pipeline) and not force:
        print(
            f"\n[{cell_number}/{total_cells}] SKIP (cached): "
            f"{pipeline} on {topic.id!r} — metadata.json exists",
            flush=True,
        )
        return

    print(
        f"\n[{cell_number}/{total_cells}] RUN: {pipeline} on {topic.id!r}",
        flush=True,
    )
    started = datetime.now(UTC)
    result = await _invoke(pipeline, topic.topic)
    completed = datetime.now(UTC)
    duration_s = (completed - started).total_seconds()

    if isinstance(result, BriefingSuccess):
        artifact = CellArtifact(
            run_id=run_id,
            topic_id=topic.id,
            topic=topic.topic,
            pipeline=pipeline,
            status="success",
            run_started_at=started.isoformat(),
            run_completed_at=completed.isoformat(),
            duration_s=duration_s,
            thread_id=result.thread_id,
            source_urls=list(result.source_urls),
            briefing_chars=len(result.briefing_markdown),
        )
        write_cell(artifact, result.briefing_markdown)
        print(
            f"   ✓ {duration_s:.1f}s, {len(result.source_urls)} sources, "
            f"{len(result.briefing_markdown)} chars",
            flush=True,
        )
    elif isinstance(result, BriefingFailure):
        artifact = CellArtifact(
            run_id=run_id,
            topic_id=topic.id,
            topic=topic.topic,
            pipeline=pipeline,
            status="failure",
            run_started_at=started.isoformat(),
            run_completed_at=completed.isoformat(),
            duration_s=duration_s,
            thread_id=result.thread_id,
            failure_reason=result.reason,
            failure_message=result.message,
        )
        # On failure we still write a placeholder briefing.md pointing
        # at the metadata, so the results tree is readable by `ls`.
        placeholder = (
            f"# {topic.topic}\n\n"
            f"_Run failed with reason `{result.reason}`. "
            f"See metadata.json for details; see LangSmith trace "
            f"`{result.thread_id}` for the full execution record._\n\n"
            f"> {result.message[:500]}\n"
        )
        write_cell(artifact, placeholder)
        print(
            f"   ✗ FAILURE ({result.reason}): {result.message[:120]!r}",
            flush=True,
        )
    else:
        # BriefingResult is a Union[BriefingSuccess, BriefingFailure];
        # this branch means the contract broke. Fail loud — not data.
        raise TypeError(
            f"run_briefing returned an unexpected type: {type(result).__name__}"
        )


async def main(args: argparse.Namespace) -> None:
    _activate_cache(CACHE_DIR)

    topics = load_eval_topics(EVAL_SET_PATH, filter_ids=args.topics)
    pipelines = tuple(args.pipelines)

    plan = [
        (topic, pipeline)
        for pipeline in pipelines
        for topic in topics
    ]
    total = len(plan)

    print(f"Harness run: run_id={args.run_id!r}")
    print(f"Topics: {[t.id for t in topics]}")
    print(f"Pipelines: {list(pipelines)}")
    print(f"Cache dir: {CACHE_DIR.relative_to(REPO_ROOT)}")
    print(f"Results dir: {(RESULTS_DIR / args.run_id).relative_to(REPO_ROOT)}/")
    if os.environ.get("TAVILY_CACHE_REFRESH", "").lower() in {"1", "true", "yes"}:
        print("TAVILY_CACHE_REFRESH is set — all queries will re-fetch.")
    if args.force:
        print("--force passed: re-running every cell.")
    else:
        print("Default: skip cells with existing metadata.json.")

    script_started = datetime.now(UTC)

    for i, (topic, pipeline) in enumerate(plan, start=1):
        await run_cell(
            run_id=args.run_id,
            topic=topic,
            pipeline=pipeline,
            force=args.force,
            cell_number=i,
            total_cells=total,
        )

    total_duration = (datetime.now(UTC) - script_started).total_seconds()
    print(f"\nDone. Total harness runtime: {total_duration:.1f}s")
    print(f"Results -> {(RESULTS_DIR / args.run_id).relative_to(REPO_ROOT)}/")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--run-id",
        default="current",
        help="Subdirectory name under evals/results/ (default: 'current').",
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        default=None,
        help="Topic IDs to run (default: all in eval_set.json).",
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        default=list(PIPELINES),
        choices=list(PIPELINES),
        help="Which pipelines to run (default: both).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run every cell, even if metadata.json already exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\n\nInterrupted. Partial results remain on disk — re-run to continue.")
        sys.exit(130)