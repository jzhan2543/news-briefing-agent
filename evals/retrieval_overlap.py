"""
Retrieval-overlap invariant for cross-pipeline comparison.

Background (see FAILURES.md, "Tavily cache: Retrieval not actually pinned
across pipelines"): the Tavily cache pins replay-determinism per query
string, but the researcher sub-agent in the single-agent and multi-agent
pipelines rephrases queries based on upstream state, so the two pipelines
end up with different `web_search` tool-call arguments and therefore
different article sets. The harness was implicitly claiming cross-pipeline
retrieval equivalence; the cache only supports per-query replay.

This module is the harness-level assertion that catches that class of
divergence. It reads `metadata.json` for both pipelines of each topic
under a run_id, normalizes URLs, and computes the Jaccard similarity
of the two `source_urls` sets. Below a threshold, cross-pipeline
metric comparisons on that topic need the retrieval-overlap caveat
appended when reported.

Design notes:

1. URL normalization reuses `source_tiers._url_to_key` — lowercase host,
   strip leading `www.`, keep path. Two URLs that differ only in `www.`
   prefix or trailing slash count as the same source for overlap purposes.
   The goal is measuring "did the pipelines see the same content?", not
   "did they receive byte-identical strings?"

2. Threshold of 0.7 is a soft gate, not a hard fail. The module reports;
   it doesn't raise. Downstream metric reporting is what consults the
   overlap field to decide whether a pairwise comparison is safe.

3. No LLM calls, no network. Reads local JSON files. Runs in
   milliseconds per topic. Re-runs are free.

4. Output is `retrieval_overlap.json` written to the run directory,
   parallel to the per-cell scores. Aggregation consumers (the Day 7
   comparison table) load this alongside source_tiers output.

Usage:
    PYTHONPATH=. python evals/retrieval_overlap.py
    PYTHONPATH=. python evals/retrieval_overlap.py --topics iphone_launch_rumors
    PYTHONPATH=. python evals/retrieval_overlap.py --threshold 0.5
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from evals.source_tiers import _url_to_key  # reuse normalization


# ---------------------------------------------------------------------------
# Paths — mirror source_tiers.py
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "evals" / "results"
DEFAULT_RUN_ID = "current"
DEFAULT_THRESHOLD = 0.7


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------


@dataclass
class TopicOverlap:
    """Per-topic retrieval-overlap result."""

    topic_id: str
    single_agent_count: int
    multi_agent_count: int
    intersection_count: int
    union_count: int
    jaccard: float
    passes_threshold: bool
    threshold: float
    only_single_agent: list[str]
    only_multi_agent: list[str]


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _normalize_url(url: str) -> str:
    """Normalize a URL to a host-path key for overlap comparison.

    Reuses the same normalization as the source-tier classifier so a
    cross-pipeline URL match and a tier lookup agree on what counts
    as "the same URL."
    """
    _, host_path = _url_to_key(url)
    return host_path


def compute_overlap(
    single_urls: list[str],
    multi_urls: list[str],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[float, set[str], set[str], set[str]]:
    """Return (jaccard, intersection, only_single, only_multi).

    Edge cases:
    - Both empty → jaccard=1.0 (vacuously identical). This only happens
      on failed cells and should be caught upstream; we return 1.0 here
      rather than NaN because the caller can still serialize it.
    - One empty, one non-empty → jaccard=0.0.
    """
    single_set = {_normalize_url(u) for u in single_urls}
    multi_set = {_normalize_url(u) for u in multi_urls}

    intersection = single_set & multi_set
    union = single_set | multi_set

    if not union:
        # Both pipelines retrieved zero URLs. Vacuously identical.
        return 1.0, set(), set(), set()

    jaccard = len(intersection) / len(union)
    only_single = single_set - multi_set
    only_multi = multi_set - single_set
    return jaccard, intersection, only_single, only_multi


def _load_source_urls(metadata_path: Path) -> list[str]:
    """Load source_urls from a metadata.json. Raises if the file is missing."""
    with metadata_path.open() as f:
        metadata = json.load(f)
    urls = metadata.get("source_urls", [])
    if not isinstance(urls, list):
        raise ValueError(
            f"{metadata_path}: source_urls is not a list (got {type(urls).__name__})"
        )
    return urls


def score_topic(
    run_id: str,
    topic_id: str,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> TopicOverlap:
    """Compute the retrieval-overlap result for one topic under a run_id."""
    topic_dir = RESULTS_DIR / run_id / topic_id
    single_meta = topic_dir / "single_agent" / "metadata.json"
    multi_meta = topic_dir / "multi_agent" / "metadata.json"

    single_urls = _load_source_urls(single_meta)
    multi_urls = _load_source_urls(multi_meta)

    jaccard, _intersection, only_single, only_multi = compute_overlap(
        single_urls, multi_urls, threshold=threshold
    )
    union_count = len({_normalize_url(u) for u in single_urls} | {_normalize_url(u) for u in multi_urls})
    intersection_count = len({_normalize_url(u) for u in single_urls} & {_normalize_url(u) for u in multi_urls})

    return TopicOverlap(
        topic_id=topic_id,
        single_agent_count=len(single_urls),
        multi_agent_count=len(multi_urls),
        intersection_count=intersection_count,
        union_count=union_count,
        jaccard=jaccard,
        passes_threshold=jaccard >= threshold,
        threshold=threshold,
        only_single_agent=sorted(only_single),
        only_multi_agent=sorted(only_multi),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _iter_topics(run_id: str, topic_filter: set[str] | None) -> list[str]:
    """List topic directories under a run_id that have both pipelines present."""
    run_dir = RESULTS_DIR / run_id
    if not run_dir.is_dir():
        return []
    topics = []
    for topic_dir in sorted(run_dir.iterdir()):
        if not topic_dir.is_dir():
            continue
        if topic_filter and topic_dir.name not in topic_filter:
            continue
        single_meta = topic_dir / "single_agent" / "metadata.json"
        multi_meta = topic_dir / "multi_agent" / "metadata.json"
        if single_meta.is_file() and multi_meta.is_file():
            topics.append(topic_dir.name)
    return topics


def _format_table(results: list[TopicOverlap]) -> str:
    """Render a small fixed-width summary table."""
    header = f"{'topic':<24} {'|A|':>4} {'|B|':>4} {'|A∩B|':>6} {'|A∪B|':>6} {'jacc':>6} {'pass':>5}"
    sep = "-" * len(header)
    rows = [header, sep]
    for r in results:
        flag = "✓" if r.passes_threshold else "✗"
        rows.append(
            f"{r.topic_id:<24} {r.single_agent_count:>4} {r.multi_agent_count:>4} "
            f"{r.intersection_count:>6} {r.union_count:>6} {r.jaccard:>6.2f} {flag:>5}"
        )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--topics",
        nargs="*",
        help="Only score these topic IDs. Default: all topics under the run.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Jaccard threshold for passing. Default {DEFAULT_THRESHOLD}.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write retrieval_overlap.json to the run directory.",
    )
    args = parser.parse_args()

    topic_filter = set(args.topics) if args.topics else None
    topics = _iter_topics(args.run_id, topic_filter)
    if not topics:
        print(f"No topics found under run_id={args.run_id}")
        return

    results = [score_topic(args.run_id, t, threshold=args.threshold) for t in topics]
    print(_format_table(results))

    failed = [r for r in results if not r.passes_threshold]
    if failed:
        print()
        print(f"Below threshold ({args.threshold}):")
        for r in failed:
            print(f"  {r.topic_id}: jaccard={r.jaccard:.2f}")
            if r.only_single_agent:
                print(f"    only in single_agent ({len(r.only_single_agent)}):")
                for url in r.only_single_agent[:5]:
                    print(f"      {url}")
                if len(r.only_single_agent) > 5:
                    print(f"      ... and {len(r.only_single_agent) - 5} more")
            if r.only_multi_agent:
                print(f"    only in multi_agent ({len(r.only_multi_agent)}):")
                for url in r.only_multi_agent[:5]:
                    print(f"      {url}")
                if len(r.only_multi_agent) > 5:
                    print(f"      ... and {len(r.only_multi_agent) - 5} more")

    if args.write:
        out = {
            "run_id": args.run_id,
            "threshold": args.threshold,
            "topics": {r.topic_id: asdict(r) for r in results},
        }
        out_path = RESULTS_DIR / args.run_id / "retrieval_overlap.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()