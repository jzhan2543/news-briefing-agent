"""
Aggregation pass — combines per-cell metrics and per-run retrieval overlap
into one comparison table.

Reads (all pre-written as JSON):
- <run>/<topic>/<pipeline>/metadata.json     — from run_evals.py
- <run>/<topic>/<pipeline>/source_tiers.json — from source_tiers.py --write-scores
- <run>/<topic>/<pipeline>/llm_judge.json    — from llm_judge.py --write-scores
- <run>/retrieval_overlap.json               — from retrieval_overlap.py --write

Writes:
- <run>/comparison.csv — one row per cell, 15 columns, machine-readable
- <run>/comparison.md  — grouped by topic, single vs multi side-by-side,
                         retrieval-pinning caveat per topic

Design notes:

1. JSON-only. This module doesn't import source_tiers or llm_judge. Every
   metric module writes its own JSON; the aggregator reads them. Matches
   the "harness produces artifacts, metrics are a separate pass over the
   artifacts" principle from Day 6. Lets any of the four producers be
   regenerated independently without touching this module.

2. No composite score. The table shows each metric in its own column.
   Collapsing tier1_share, claim_support_rate, and utilization_score
   into one number would hide that they measure different failure
   modes (see DESIGN.md §7 and the two-part-judge rationale in
   evals/llm_judge.py).

3. Caveats inline. Below-threshold Jaccard cells get an asterisk in
   both the CSV and the markdown header. The retrieval-pinning status
   is the load-bearing caveat — pairwise comparisons on those topics
   mix architectural behavior with retrieval divergence, and that has
   to be visible at read-time, not relegated to a footnote.

4. Missing sources are explicit. If llm_judge.json doesn't exist for a
   cell, the CSV row still renders with blanks and the markdown marks
   the missing metrics. Silent zero-filling would masquerade as real
   scores.

Usage:
    # Regenerate all metric artifacts then aggregate
    PYTHONPATH=. python evals/source_tiers.py --write-scores
    PYTHONPATH=. python evals/llm_judge.py --write-scores
    PYTHONPATH=. python evals/retrieval_overlap.py --write
    PYTHONPATH=. python evals/aggregate.py

    # Narrow to a single run
    PYTHONPATH=. python evals/aggregate.py --run-id current
"""

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "evals" / "results"
DEFAULT_RUN_ID = "current"
PIPELINES = ("single_agent", "multi_agent")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class CellRow:
    """All the fields the aggregator pulls into one row per cell.

    Values are `None` where the source file is missing, which renders
    as blank in CSV and `—` in markdown.
    """
    topic_id: str
    pipeline: str
    # From metadata.json
    duration_s: float | None = None
    briefing_chars: int | None = None
    # From source_tiers.json
    citations_total: int | None = None
    tier1_share: float | None = None
    tier3_share: float | None = None
    ugc_flag: int | None = None
    retrieval_total: int | None = None
    retrieval_to_citation_rate: float | None = None
    # From llm_judge.json
    claim_support_rate: float | None = None
    claims_unverifiable: int | None = None
    utilization_score: float | None = None
    omissions_high: int | None = None
    # From retrieval_overlap.json (per topic)
    jaccard: float | None = None
    retrieval_pinned: bool | None = None
    missing_sources: list[str] = field(default_factory=list)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _iter_cells(run_id: str, topic_filter: set[str] | None) -> list[tuple[str, str]]:
    """Yield (topic_id, pipeline) pairs with at least metadata.json present."""
    run_dir = RESULTS_DIR / run_id
    if not run_dir.is_dir():
        return []
    pairs = []
    for topic_dir in sorted(run_dir.iterdir()):
        if not topic_dir.is_dir():
            continue
        if topic_filter and topic_dir.name not in topic_filter:
            continue
        for pipeline in PIPELINES:
            if (topic_dir / pipeline / "metadata.json").is_file():
                pairs.append((topic_dir.name, pipeline))
    return pairs


def collect_rows(run_id: str, topic_filter: set[str] | None = None) -> list[CellRow]:
    """Read all JSON artifacts under a run_id and produce one CellRow per cell."""
    run_dir = RESULTS_DIR / run_id
    # Per-run artifact: retrieval_overlap
    overlap_data = _load_json(run_dir / "retrieval_overlap.json") or {}
    overlap_by_topic = overlap_data.get("topics", {})

    rows: list[CellRow] = []
    for topic_id, pipeline in _iter_cells(run_id, topic_filter):
        cell_dir = run_dir / topic_id / pipeline
        row = CellRow(topic_id=topic_id, pipeline=pipeline)

        metadata = _load_json(cell_dir / "metadata.json")
        if metadata:
            row.duration_s = metadata.get("duration_s")
            row.briefing_chars = metadata.get("briefing_chars")
        else:
            row.missing_sources.append("metadata")

        tiers = _load_json(cell_dir / "source_tiers.json")
        if tiers:
            row.citations_total = tiers.get("citations_total")
            row.tier1_share = tiers.get("tier1_citation_share")
            row.tier3_share = tiers.get("tier3_citation_share")
            row.ugc_flag = tiers.get("ugc_flag")
            row.retrieval_total = tiers.get("retrieval_total")
            row.retrieval_to_citation_rate = tiers.get("retrieval_to_citation_rate")
        else:
            row.missing_sources.append("source_tiers")

        judge = _load_json(cell_dir / "llm_judge.json")
        if judge:
            row.claim_support_rate = judge.get("claim_support_rate")
            row.claims_unverifiable = judge.get("claims_unverifiable")
            row.utilization_score = judge.get("retrieval_utilization_score")
            row.omissions_high = judge.get("omissions_high")
        else:
            row.missing_sources.append("llm_judge")

        topic_overlap = overlap_by_topic.get(topic_id)
        if topic_overlap:
            row.jaccard = topic_overlap.get("jaccard")
            row.retrieval_pinned = topic_overlap.get("passes_threshold")

        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "topic_id",
    "pipeline",
    "duration_s",
    "briefing_chars",
    "citations_total",
    "tier1_share",
    "tier3_share",
    "ugc_flag",
    "retrieval_total",
    "retrieval_to_citation_rate",
    "claim_support_rate",
    "claims_unverifiable",
    "utilization_score",
    "omissions_high",
    "jaccard",
    "retrieval_pinned",
]


def _fmt_csv_value(v: Any) -> str:
    """Format a value for CSV — empty string for None, ratios at 4 decimals."""
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, bool):
        return "Y" if v else "N"
    return str(v)


def write_csv(rows: list[CellRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for r in rows:
            writer.writerow([_fmt_csv_value(getattr(r, c, None)) for c in CSV_COLUMNS])


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------


def _fmt_pct(v: float | None) -> str:
    return f"{v * 100:.0f}%" if v is not None else "—"


def _fmt_float(v: float | None, places: int = 2) -> str:
    return f"{v:.{places}f}" if v is not None else "—"


def _fmt_int(v: int | None) -> str:
    return str(v) if v is not None else "—"


def _fmt_ugc(v: int | None) -> str:
    if v is None:
        return "—"
    return "⚠ yes" if v else "no"


def _pinning_header(topic_id: str, jaccard: float | None, pinned: bool | None) -> str:
    if pinned is None:
        return f"### {topic_id} — retrieval-pinning: not computed"
    if pinned:
        return f"### {topic_id} — retrieval pinned ✓ (Jaccard {jaccard:.2f})"
    return (
        f"### {topic_id} — ⚠ retrieval NOT pinned (Jaccard {jaccard:.2f}) — "
        f"pairwise deltas on this topic mix architecture with retrieval divergence"
    )


def write_markdown(rows: list[CellRow], path: Path, run_id: str) -> None:
    """Write a human-readable comparison table grouped by topic."""
    by_topic: dict[str, dict[str, CellRow]] = {}
    for r in rows:
        by_topic.setdefault(r.topic_id, {})[r.pipeline] = r

    lines: list[str] = []
    lines.append(f"# Comparison table — run `{run_id}`")
    lines.append("")
    lines.append(
        "Per-topic metric comparison between single-agent and multi-agent pipelines. "
        "Metrics kept separate (no composite score) because each measures a "
        "different failure mode. Retrieval-pinning status annotates each topic — "
        "pairwise comparisons on unpinned topics mix architectural behavior with "
        "retrieval divergence and should be treated with caution."
    )
    lines.append("")
    lines.append("**Column guide:**")
    lines.append("")
    lines.append("- `chars` — briefing length. Single-agent tends longer; Critic trims.")
    lines.append("- `cites` — in-text `[source: URL]` citations.")
    lines.append("- `T1 / T3` — citation share to tier-1 (reputable) / tier-3 (low-quality) domains, category-conditional.")
    lines.append("- `UGC` — ⚠ if any citation is user-generated-content (LinkedIn posts, YouTube, etc.)")
    lines.append("- `retr / r→c` — retrieved URL count / fraction of retrieved URLs that actually get cited.")
    lines.append("- `supp%` — per-claim support rate from the Haiku Part 1 judge.")
    lines.append("- `?` — claims citing URLs not in the Tavily cache (possible fabricated URL or cache miss).")
    lines.append("- `util` — retrieval utilization score from the Sonnet Part 2 judge. 1 − (high-importance omissions / 5). Three-value ceiling on this run: {0.60, 0.80, 1.00}.")
    lines.append("- `H` — high-importance omissions flagged by Part 2.")
    lines.append("")

    for topic_id in sorted(by_topic.keys()):
        pair = by_topic[topic_id]
        sample = next(iter(pair.values()))
        lines.append(_pinning_header(topic_id, sample.jaccard, sample.retrieval_pinned))
        lines.append("")
        lines.append(
            "| pipeline | chars | cites | T1 | T3 | UGC | retr | r→c | supp% | ? | util | H |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|"
        )
        for pipeline in PIPELINES:
            r = pair.get(pipeline)
            if r is None:
                lines.append(f"| {pipeline} | — | — | — | — | — | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {r.pipeline} "
                f"| {_fmt_int(r.briefing_chars)} "
                f"| {_fmt_int(r.citations_total)} "
                f"| {_fmt_pct(r.tier1_share)} "
                f"| {_fmt_pct(r.tier3_share)} "
                f"| {_fmt_ugc(r.ugc_flag)} "
                f"| {_fmt_int(r.retrieval_total)} "
                f"| {_fmt_pct(r.retrieval_to_citation_rate)} "
                f"| {_fmt_pct(r.claim_support_rate)} "
                f"| {_fmt_int(r.claims_unverifiable)} "
                f"| {_fmt_float(r.utilization_score)} "
                f"| {_fmt_int(r.omissions_high)} |"
            )
            if r.missing_sources:
                lines.append(
                    f"<!-- {pipeline}: missing {', '.join(r.missing_sources)} -->"
                )
        lines.append("")

    # Footer: data sources
    lines.append("---")
    lines.append("")
    lines.append("**Data sources per cell (all JSON on disk):**")
    lines.append("")
    lines.append("- `metadata.json` — from `run_evals.py` (pipeline output)")
    lines.append("- `source_tiers.json` — from `source_tiers.py --write-scores`")
    lines.append("- `llm_judge.json` — from `llm_judge.py --write-scores`")
    lines.append("- `retrieval_overlap.json` (per run) — from `retrieval_overlap.py --write`")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-cell metric JSON into comparison.csv + comparison.md. "
            "Run source_tiers.py --write-scores, llm_judge.py --write-scores, "
            "and retrieval_overlap.py --write first."
        )
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--topics", nargs="*", default=None)
    args = parser.parse_args()

    topic_filter = set(args.topics) if args.topics else None
    rows = collect_rows(args.run_id, topic_filter)
    if not rows:
        print(f"No cells found under {RESULTS_DIR / args.run_id}")
        return

    run_dir = RESULTS_DIR / args.run_id
    csv_path = run_dir / "comparison.csv"
    md_path = run_dir / "comparison.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path, args.run_id)

    # Terminal summary
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"  {len(rows)} cells, {len({r.topic_id for r in rows})} topics")
    missing_any = [r for r in rows if r.missing_sources]
    if missing_any:
        print(f"  WARNING: {len(missing_any)} cells have missing source files:")
        for r in missing_any:
            print(f"    {r.topic_id}/{r.pipeline}: missing {', '.join(r.missing_sources)}")


if __name__ == "__main__":
    main()