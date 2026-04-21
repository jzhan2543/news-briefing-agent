"""
Faithfulness judge for briefing cells.

Two-part LLM-as-judge evaluation that runs on the per-cell artifacts
written by `run_evals.py`:

Part 1 — claim support.
    For each [source: URL] citation in the briefing, extract the
    supporting sentence, retrieve the corresponding article snippet
    from the Tavily cache, and ask the judge whether the snippet
    supports the claim. Aggregates to `claim_support_rate`.

Part 2 — retrieval utilization.
    Give the judge the full briefing + every retrieved article's
    title/snippet (both cited and non-cited), and ask it to identify
    substantive facts or perspectives present in the retrieved corpus
    but absent from the briefing. Rate each omission's importance.
    Aggregates to `retrieval_utilization_score = 1 - (high / cap)`
    where cap is the max number of omissions the judge is asked to
    surface (default 5).

Why two parts, not one:

Part 1 alone is the obvious faithfulness check. It catches the
FAILURES.md Formatter-quote-attribution case: "Fed statement noted
'uncertain'" cited to CNBC — claim text vs. cited snippet tells you
whether the attribution is real.

But Part 1 misses a different failure mode we observed on the
EU AI Act pair: multi-agent dropped the EDRi civil-society
open-letter coverage *entirely*. Every surviving claim was
well-supported by its source — Part 1 would score the briefing
cleanly. The failure is *what's absent*, not what's wrong. Part 2
is the test for that.

Model choice:

Defaults to claude-haiku for Part 1 (structured claim-verification;
self-preference bias is low because the judge isn't grading writing
quality, it's doing constrained yes/no/partial support classification)
and claude-sonnet for Part 2 (requires reading a larger context and
making judgment calls about "substantive" that Haiku has been flaky
on in our small-sample checks). Both overridable via CLI; the point
of splitting is to let sweeps reason about them independently.

Cost & caching:

Every judge call is wrapped in a disk cache keyed on
(model, part, payload_sha). Re-running after a metric tweak costs
nothing unless the briefing content or judge model changes. Same
record/replay pattern as src/tools/tavily_cache.py — file-per-call,
human-inspectable.

What this module does NOT do:

- It does not re-run the pipeline. It reads briefing.md, metadata.json,
  and the Tavily cache as-of-now.
- It does not auto-fix flagged issues. Afternoon scope is
  measurement, not tuning.
- It does not ship a "overall faithfulness score." Part 1 and
  Part 2 measure different things; collapsing them into one number
  would hide the signal that splits them.

Usage:
    # Score every cell under a run_id, print a table, cache judge calls
    PYTHONPATH=. python evals/llm_judge.py

    # Narrow to one topic
    PYTHONPATH=. python evals/llm_judge.py --topics eu_ai_act_enforcement

    # Skip Part 2 (the expensive part) while iterating on Part 1 prompts
    PYTHONPATH=. python evals/llm_judge.py --skip-part2

    # Use a different judge model (sweep)
    PYTHONPATH=. python evals/llm_judge.py --part1-model claude-sonnet-4-5

    # Write per-cell judge outputs alongside metadata.json
    PYTHONPATH=. python evals/llm_judge.py --write-scores
"""

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

# Lazy imports for the LLM client — keeps this module importable in
# environments without langchain-anthropic for unit-testing the pure
# functions (prompt assembly, cache keys, aggregation).
try:
    from langchain_anthropic import ChatAnthropic  # type: ignore
except ImportError:  # pragma: no cover
    ChatAnthropic = None  # type: ignore


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "evals" / "results"
TAVILY_CACHE_DIR = REPO_ROOT / "evals" / "cache" / "tavily"
JUDGE_CACHE_DIR = REPO_ROOT / "evals" / "cache" / "judge"

DEFAULT_RUN_ID = "current"
PIPELINES = ("single_agent", "multi_agent")

DEFAULT_PART1_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_PART2_MODEL = "claude-sonnet-4-5"

# How many omissions to surface in Part 2. Denominator for the
# utilization score. Set deliberately small — the point is to catch
# the 1-3 most load-bearing omissions, not to enumerate every fact
# not in the briefing.
PART2_OMISSION_CAP = 5


# ---------------------------------------------------------------------------
# Citation parsing with sentence extraction
# ---------------------------------------------------------------------------

_CITATION_RE = re.compile(r"\[source:\s*(https?://[^\s\]]+)")


@dataclass(frozen=True)
class Citation:
    """One in-text citation: the URL cited and the sentence it's in.

    The sentence is the claim-under-evaluation for Part 1. Extracted
    by walking back from the citation marker to the nearest prior
    sentence boundary. Imperfect — long compound sentences give a
    longer "claim" than needed — but precision here is less important
    than determinism, so the heuristic is deliberately simple.
    """
    url: str
    sentence: str


_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")


def extract_citations_with_sentences(briefing: str) -> list[Citation]:
    """Return one Citation per [source: URL] marker.

    The "sentence" is the text from the previous sentence boundary up
    to and including the citation marker. If the same URL is cited N
    times in N different sentences, N Citation records are returned —
    duplicates are meaningful signal.
    """
    citations: list[Citation] = []
    for match in _CITATION_RE.finditer(briefing):
        url = match.group(1)
        citation_end = match.end()
        # Find the closing ] for this citation span.
        close_bracket = briefing.find("]", citation_end)
        span_end = close_bracket + 1 if close_bracket != -1 else citation_end

        # Walk back to the prior sentence boundary.
        head = briefing[:match.start()]
        boundaries = list(_SENTENCE_BOUNDARY_RE.finditer(head))
        sentence_start = boundaries[-1].end() if boundaries else 0
        sentence = briefing[sentence_start:span_end].strip()
        citations.append(Citation(url=url, sentence=sentence))
    return citations


# ---------------------------------------------------------------------------
# Tavily cache → URL → article content index
#
# Every cached response is a file storing {"query": ..., "results": [...]}.
# Results contain url/title/content. We walk all files and build a
# url → (title, content) map. Same URL can appear under multiple queries;
# we keep the first non-empty content seen.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArticleSnippet:
    url: str
    title: str
    content: str


def load_tavily_cache_index(cache_dir: Path = TAVILY_CACHE_DIR) -> dict[str, ArticleSnippet]:
    """Build a url → ArticleSnippet map from the on-disk Tavily cache.

    Returns an empty dict if cache_dir doesn't exist. Does not raise on
    malformed cache files — those are skipped silently, matching
    tavily_cache._read_entry tolerance.
    """
    index: dict[str, ArticleSnippet] = {}
    if not cache_dir.exists():
        return index
    for path in sorted(cache_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        results = payload.get("results")
        if not isinstance(results, list):
            continue
        for r in results:
            url = r.get("url")
            if not url or url in index:
                continue
            content = r.get("content") or r.get("raw_content") or ""
            index[url] = ArticleSnippet(
                url=url,
                title=r.get("title") or "",
                content=content,
            )
    return index


# ---------------------------------------------------------------------------
# Judge-call cache
#
# Keyed on (model, part_tag, payload_sha). payload_sha is a hash of
# whatever input the judge is seeing — claim+snippet for Part 1, full
# briefing+corpus for Part 2. Re-running with an unchanged briefing
# and an unchanged model gets zero new LLM calls.
# ---------------------------------------------------------------------------

def _judge_cache_path(model: str, part: str, payload: str) -> Path:
    sha = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    safe_model = model.replace("/", "_").replace(":", "_")
    return JUDGE_CACHE_DIR / f"{part}_{safe_model}_{sha}.json"


def _read_judge_cache(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_judge_cache(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# Part 1: per-claim support classifier
# ---------------------------------------------------------------------------

_PART1_SYSTEM = """\
You are evaluating whether a specific claim in a news briefing is
supported by the source article it cites. Your job is classification,
not editing.

For each (claim, source snippet) pair, return one of:
- "supported":   The snippet directly supports the claim, including
                 any quoted phrasings, numbers, or attributions.
- "partial":     The snippet is consistent with the claim but the
                 claim adds detail, specificity, or interpretation
                 not present in the snippet. Includes cases where
                 the briefing generalizes a narrower source.
- "unsupported": The snippet does not support the claim, contradicts
                 it, or is about a different subject.
- "unverifiable": The snippet is empty, malformed, or too short to
                 evaluate.

For quoted text in the claim (text inside single or double quotes
attributed to a source), require a verbatim-or-near-verbatim match
in the snippet to classify "supported". If the quoted string does
not appear in the snippet, classify "unsupported" even if the
paraphrased meaning is present — this catches fabricated
attributions.
"""

_PART1_USER_TEMPLATE = """\
CLAIM:
{claim}

CITED SOURCE URL: {url}
CITED SOURCE TITLE: {title}

SOURCE SNIPPET:
{snippet}

Return a single JSON object with keys:
  verdict: one of "supported" | "partial" | "unsupported" | "unverifiable"
  reason:  one sentence, max 40 words, explaining the verdict. If
           "unsupported" because of a quote mismatch, quote the
           exact string from the claim that's not in the snippet.
"""


@dataclass(frozen=True)
class ClaimVerdict:
    url: str
    claim: str
    verdict: str  # supported | partial | unsupported | unverifiable
    reason: str


def judge_claim(
    citation: Citation,
    snippet: ArticleSnippet | None,
    model: str,
    llm_call: Callable[[str, str, str], dict],
) -> ClaimVerdict:
    """Classify one claim against its cited snippet.

    `llm_call(system, user, model) -> {"verdict":..., "reason":...}`
    is injected so this function is testable without network. The
    default production binding is `_default_llm_call` below.
    """
    if snippet is None or not snippet.content.strip():
        return ClaimVerdict(
            url=citation.url,
            claim=citation.sentence,
            verdict="unverifiable",
            reason="No cached snippet available for the cited URL.",
        )

    user = _PART1_USER_TEMPLATE.format(
        claim=citation.sentence,
        url=citation.url,
        title=snippet.title,
        snippet=snippet.content,
    )
    response = llm_call(_PART1_SYSTEM, user, model)
    return ClaimVerdict(
        url=citation.url,
        claim=citation.sentence,
        verdict=response.get("verdict", "unverifiable"),
        reason=response.get("reason", ""),
    )


# ---------------------------------------------------------------------------
# Part 2: retrieval utilization
# ---------------------------------------------------------------------------

_PART2_SYSTEM = """\
You are evaluating whether a news briefing made good use of the
articles that were retrieved for it. You will be given:

1. The briefing as published.
2. The full list of retrieved articles (title + snippet each),
   including those the briefing cites and those it does not.

Your job: identify up to {cap} substantive facts or perspectives
that are present in the retrieved corpus but absent from the
briefing. For each, rate its importance:

- "high":   A fact or perspective whose absence changes the
            briefing's framing or omits a load-bearing piece of
            evidence (e.g. a civil-society finding that directly
            contradicts the briefing's tone, a primary-source
            number on a topic the briefing discusses in vaguer
            terms, a named stakeholder whose view is entirely
            absent).
- "medium": A fact that adds useful context but isn't
            framing-changing. The briefing would be better for
            including it.
- "low":    A detail that's fine to omit — redundant with cited
            content, marginal relevance, or appropriately tight
            scoping.

Scoring discipline:

- Only flag omissions where the content IS in a retrieved article.
  Do not flag general topic gaps ("the briefing should have talked
  about X") unless X appears in the retrieved corpus.
- Ignore stylistic or length preferences. A shorter briefing that
  captures the substance is not penalized.
- If the briefing is already comprehensive, return an empty list.
  Do not invent "high" omissions to fill the cap.
"""

_PART2_USER_TEMPLATE = """\
BRIEFING:
{briefing}

RETRIEVED ARTICLES ({n_articles} total):
{articles_block}

Return a single JSON object with keys:
  omissions: list of objects, each with:
    - summary: the missing fact/perspective in one sentence (max 30 words)
    - source_url: the URL of the retrieved article where this appears
    - importance: "high" | "medium" | "low"
  note: optional one-sentence observation about the briefing's
        overall use of the retrieved corpus. May be empty string.
"""


@dataclass(frozen=True)
class Omission:
    summary: str
    source_url: str
    importance: str


@dataclass(frozen=True)
class UtilizationVerdict:
    omissions: tuple[Omission, ...]
    note: str


def _format_articles_block(retrieved: list[ArticleSnippet]) -> str:
    """Render retrieved articles as numbered title/url/content blocks."""
    lines: list[str] = []
    for i, a in enumerate(retrieved, start=1):
        lines.append(f"[{i}] {a.title}")
        lines.append(f"    URL: {a.url}")
        lines.append(f"    CONTENT: {a.content}")
        lines.append("")
    return "\n".join(lines)


def judge_utilization(
    briefing: str,
    retrieved: list[ArticleSnippet],
    model: str,
    llm_call: Callable[[str, str, str], dict],
) -> UtilizationVerdict:
    """Identify substantive retrieved-but-unused content."""
    if not retrieved:
        return UtilizationVerdict(omissions=(), note="No retrieved articles available.")

    system = _PART2_SYSTEM.format(cap=PART2_OMISSION_CAP)
    user = _PART2_USER_TEMPLATE.format(
        briefing=briefing,
        n_articles=len(retrieved),
        articles_block=_format_articles_block(retrieved),
    )
    response = llm_call(system, user, model)
    raw_omissions = response.get("omissions", []) or []
    omissions = tuple(
        Omission(
            summary=o.get("summary", ""),
            source_url=o.get("source_url", ""),
            importance=o.get("importance", "low"),
        )
        for o in raw_omissions[:PART2_OMISSION_CAP]
    )
    return UtilizationVerdict(omissions=omissions, note=response.get("note", ""))


# ---------------------------------------------------------------------------
# LLM call — default production binding
#
# Wraps ChatAnthropic with a disk cache. The judge functions above take
# `llm_call` as a parameter so tests can inject a stub without touching
# this code path at all.
# ---------------------------------------------------------------------------

def _default_llm_call(system: str, user: str, model: str) -> dict:
    """Call ChatAnthropic, parse JSON from the response, cache on disk.

    part_tag is derived from the system prompt's opening so Part 1 and
    Part 2 entries stay separate in the cache directory.
    """
    if ChatAnthropic is None:
        raise RuntimeError(
            "langchain_anthropic is not installed; install it or inject "
            "a stub llm_call into the judge functions."
        )
    part_tag = "part1" if "supported by the source article" in system else "part2"
    payload_for_key = f"{system}\n---\n{user}"
    cache_path = _judge_cache_path(model, part_tag, payload_for_key)
    cached = _read_judge_cache(cache_path)
    if cached is not None:
        return cached

    llm = ChatAnthropic(model=model, temperature=0.0, max_tokens=2048)
    response = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])
    text = response.content if isinstance(response.content, str) else str(response.content)
    parsed = _parse_json_from_response(text)
    _write_judge_cache(cache_path, parsed)
    return parsed


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_from_response(text: str) -> dict:
    """Extract the first JSON object from a possibly-prose-wrapped response.

    The judge prompts ask for a JSON object and Sonnet/Haiku at temp=0
    reliably comply, but occasionally wrap in ```json fences or a
    preamble. We extract the first {...} span and parse it. On parse
    failure return an empty dict — callers default verdicts to
    "unverifiable" / empty omissions, which is the conservative
    behavior (don't let a bad judge response silently score cleanly).
    """
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Per-cell orchestration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellJudgeScore:
    topic_id: str
    pipeline: str
    claims_total: int
    claims_supported: int
    claims_partial: int
    claims_unsupported: int
    claims_unverifiable: int
    claim_support_rate: float
    retrieval_utilization_score: float
    omissions_high: int
    omissions_medium: int
    omissions_low: int
    claim_verdicts: tuple[ClaimVerdict, ...] = field(default_factory=tuple)
    utilization: UtilizationVerdict | None = None


def _cell_dir(run_id: str, topic_id: str, pipeline: str) -> Path:
    return RESULTS_DIR / run_id / topic_id / pipeline


def score_cell(
    run_id: str,
    topic_id: str,
    pipeline: str,
    *,
    tavily_index: dict[str, ArticleSnippet] | None = None,
    part1_model: str = DEFAULT_PART1_MODEL,
    part2_model: str = DEFAULT_PART2_MODEL,
    skip_part2: bool = False,
    llm_call: Callable[[str, str, str], dict] = _default_llm_call,
) -> CellJudgeScore:
    """Run both judge parts on one cell, return aggregated verdicts.

    `tavily_index` is built once by the CLI entry point and passed in
    so a multi-cell run doesn't re-walk the cache directory per cell.
    """
    cell = _cell_dir(run_id, topic_id, pipeline)
    briefing = (cell / "briefing.md").read_text(encoding="utf-8")
    metadata = json.loads((cell / "metadata.json").read_text(encoding="utf-8"))

    if tavily_index is None:
        tavily_index = load_tavily_cache_index()

    # Part 1: per-claim support.
    citations = extract_citations_with_sentences(briefing)
    verdicts: list[ClaimVerdict] = [
        judge_claim(c, tavily_index.get(c.url), part1_model, llm_call)
        for c in citations
    ]
    counts = {"supported": 0, "partial": 0, "unsupported": 0, "unverifiable": 0}
    for v in verdicts:
        counts[v.verdict] = counts.get(v.verdict, 0) + 1

    # supported + 0.5 * partial, normalized by total cited claims. Partial
    # credit reflects that "consistent-but-extrapolated" is not the same
    # failure mode as unsupported. Unverifiable is in the denominator — a
    # briefing that cites URLs not in the cache is worse than one that
    # doesn't, and we want that to show.
    total_claims = len(verdicts)
    claim_support_rate = (
        (counts["supported"] + 0.5 * counts["partial"]) / total_claims
        if total_claims else 0.0
    )

    # Part 2: retrieval utilization.
    retrieved_urls = list(metadata.get("source_urls") or [])
    retrieved = [
        tavily_index[u] for u in retrieved_urls if u in tavily_index
    ]

    utilization: UtilizationVerdict | None = None
    omissions_high = omissions_medium = omissions_low = 0
    utilization_score = 1.0

    if not skip_part2 and retrieved:
        utilization = judge_utilization(briefing, retrieved, part2_model, llm_call)
        for o in utilization.omissions:
            if o.importance == "high":
                omissions_high += 1
            elif o.importance == "medium":
                omissions_medium += 1
            else:
                omissions_low += 1
        utilization_score = 1.0 - (omissions_high / PART2_OMISSION_CAP)
        utilization_score = max(0.0, min(1.0, utilization_score))

    return CellJudgeScore(
        topic_id=topic_id,
        pipeline=pipeline,
        claims_total=total_claims,
        claims_supported=counts["supported"],
        claims_partial=counts["partial"],
        claims_unsupported=counts["unsupported"],
        claims_unverifiable=counts["unverifiable"],
        claim_support_rate=claim_support_rate,
        retrieval_utilization_score=utilization_score,
        omissions_high=omissions_high,
        omissions_medium=omissions_medium,
        omissions_low=omissions_low,
        claim_verdicts=tuple(verdicts),
        utilization=utilization,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _iter_cells(run_id: str, topic_filter: set[str] | None):
    run_dir = RESULTS_DIR / run_id
    if not run_dir.exists():
        return
    for topic_dir in sorted(run_dir.iterdir()):
        if not topic_dir.is_dir():
            continue
        if topic_filter and topic_dir.name not in topic_filter:
            continue
        for pipeline in PIPELINES:
            if (topic_dir / pipeline / "metadata.json").exists():
                yield topic_dir.name, pipeline


def _format_table(scores: list[CellJudgeScore]) -> str:
    cols = [
        ("topic", 22),
        ("pipeline", 13),
        ("claims", 7),
        ("supp%", 6),
        ("S/P/U/?", 10),
        ("util", 5),
        ("H/M/L", 7),
    ]
    header = "  ".join(name.ljust(w) for name, w in cols)
    rule = "-" * len(header)
    lines = [header, rule]
    for s in scores:
        spu = f"{s.claims_supported}/{s.claims_partial}/{s.claims_unsupported}/{s.claims_unverifiable}"
        hml = f"{s.omissions_high}/{s.omissions_medium}/{s.omissions_low}"
        lines.append("  ".join([
            s.topic_id.ljust(cols[0][1]),
            s.pipeline.ljust(cols[1][1]),
            str(s.claims_total).ljust(cols[2][1]),
            f"{s.claim_support_rate:.0%}".ljust(cols[3][1]),
            spu.ljust(cols[4][1]),
            f"{s.retrieval_utilization_score:.2f}".ljust(cols[5][1]),
            hml.ljust(cols[6][1]),
        ]))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Faithfulness judge for briefing cells. Runs Part 1 "
            "(per-claim support) and Part 2 (retrieval utilization). "
            "Judge calls are cached on disk under evals/cache/judge/."
        )
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--topics", nargs="*", default=None)
    parser.add_argument("--part1-model", default=DEFAULT_PART1_MODEL)
    parser.add_argument("--part2-model", default=DEFAULT_PART2_MODEL)
    parser.add_argument("--skip-part2", action="store_true",
                        help="Run only Part 1; useful when iterating on Part 1 prompts.")
    parser.add_argument("--write-scores", action="store_true",
                        help="Write per-cell llm_judge.json next to metadata.json.")
    args = parser.parse_args()

    topic_filter = set(args.topics) if args.topics else None
    tavily_index = load_tavily_cache_index()
    scores: list[CellJudgeScore] = []

    for topic_id, pipeline in _iter_cells(args.run_id, topic_filter):
        s = score_cell(
            args.run_id, topic_id, pipeline,
            tavily_index=tavily_index,
            part1_model=args.part1_model,
            part2_model=args.part2_model,
            skip_part2=args.skip_part2,
        )
        scores.append(s)
        if args.write_scores:
            out = _cell_dir(args.run_id, topic_id, pipeline) / "llm_judge.json"
            out.write_text(json.dumps(asdict(s), indent=2, sort_keys=True, default=str))

    if not scores:
        print(
            f"No cells found under {RESULTS_DIR / args.run_id}. "
            "Run evals/run_evals.py first."
        )
        return

    print(_format_table(scores))


if __name__ == "__main__":
    main()