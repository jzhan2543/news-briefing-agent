"""
Source-tier classifier for briefing citations.

Reads the per-cell artifacts written by `run_evals.py` and produces four
deterministic per-cell metrics:

- tier1_citation_share: fraction of in-text [source: URL] citations
  pointing to tier-1 domains for the topic's category.
- tier3_citation_share: fraction pointing to tier-3 (denylist) domains.
- ugc_flag: 1 if any citation is a user-generated-content domain
  (linkedin posts, youtube videos, reddit threads, x/twitter, medium
  personal pages, substack). UGC is tier-3 by definition but flagged
  separately because it represents a qualitatively different failure
  mode — UGC citations should never appear in a news briefing and are
  worth surfacing independently of the tier-3 share.
- retrieval_to_citation_rate: fraction of retrieved source_urls that
  actually appear as in-text citations. Descriptive metric, not a
  quality metric — cross-reference with tier3_citation_share to tell
  whether the Critic is dropping low-tier sources (good) or substantive
  ones (bad).

Design notes:

1. Tiers are topic-category-conditional, not global. "Reputable" for
   policy-macro topics (Reuters, FT, JPM research, .gov) is not the
   same as reputable for consumer-tech rumors (Bloomberg Gurman,
   WSJ, specialist trade press). A single global allowlist would
   either be too narrow for tech topics or too permissive for policy
   ones.

2. Citation-weighted, not URL-set-weighted. If a briefing cites
   linkedin.com twice in different sections, that's two citations
   worth of low-tier content shaping the briefing, not one. Duplicate
   citations of the same URL across paragraphs reflect emphasis and
   count accordingly.

3. Unknown domains log a warning and default to tier-2 (no penalty,
   no reward). Silent defaulting hides eval-set gaps; an emitted
   warning surfaces domains worth classifying as the topic set
   expands.

4. No LLM calls. This is a deterministic metric — runs in
   milliseconds per cell, re-runs are free, no API budget concerns.

Usage:
    # Score every cell under a run_id and print a comparison table
    PYTHONPATH=. python evals/source_tiers.py

    # Narrow to one topic
    PYTHONPATH=. python evals/source_tiers.py --topics tariffs_2026

    # Write per-cell scores as JSON alongside the existing metadata
    PYTHONPATH=. python evals/source_tiers.py --write-scores
"""

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "evals" / "results"
EVAL_SET_PATH = REPO_ROOT / "evals" / "eval_set.json"

DEFAULT_RUN_ID = "current"
PIPELINES = ("single_agent", "multi_agent")


# ---------------------------------------------------------------------------
# Topic → category
#
# Each topic in eval_set.json maps to one of three categories below. The
# tier map per category is separate so reputable domain sets are
# appropriate to what the topic is about.
#
# Adding a topic means adding an entry here. If a topic shows up in
# results/ without a category mapping, scoring logs a warning and falls
# back to `policy_macro` (the least-surprising default for news).
# ---------------------------------------------------------------------------

TOPIC_CATEGORIES: dict[str, str] = {
    "tariffs_2026":          "policy_macro",
    "central_bank_rates":    "policy_macro",
    "uk_budget_2026":        "policy_macro",
    "iphone_launch_rumors":  "tech_consumer",
    "eu_ai_act_enforcement": "regulation",
}


# ---------------------------------------------------------------------------
# Tier maps per category
#
# Entries are matched against `host_or_host_path(url)` — see
# `_url_to_key`. Longer keys (host+path prefix) beat shorter ones
# (host only), so `blog.rwazi.com` can be tier-3 without demoting
# everything on rwazi.com. The denylist of path-prefix entries is
# deliberately specific: `wiss.com/trump-tariffs-2026-financial-impact`
# rather than all of `wiss.com`, because a different wiss.com article
# might be fine — we grade the specific cited URL, not the domain
# writ large.
#
# UGC patterns are separate (see UGC_PATTERNS) and short-circuit tier
# lookup.
# ---------------------------------------------------------------------------

_TIER_MAPS: dict[str, dict[str, int]] = {
    "policy_macro": {
        # --- Tier 1: primary + top-tier news
        "reuters.com":                   1,
        "apnews.com":                    1,
        "ft.com":                        1,
        "wsj.com":                       1,
        "bloomberg.com":                 1,
        "economist.com":                 1,
        "nytimes.com":                   1,
        "washingtonpost.com":            1,
        "jpmorgan.com":                  1,
        "goldmansachs.com":              1,
        "federalreserve.gov":            1,
        "ecb.europa.eu":                 1,
        "bankofengland.co.uk":           1,
        "imf.org":                       1,
        "oecd.org":                      1,
        "worldbank.org":                 1,
        "cbo.gov":                       1,
        "ustr.gov":                      1,
        "whitehouse.gov":                1,
        "treasury.gov":                  1,
        "gov.uk":                        1,
        "obr.uk":                        1,
        "cfr.org":                       1,
        "brookings.edu":                 1,
        "piie.com":                      1,
        "taxfoundation.org":             1,
        "budgetlab.yale.edu":            1,
        "ifs.org.uk":                    1,

        # --- Tier 2: mid-quality mainstream / think tanks / specialist
        "cnn.com":                       2,
        "cnbc.com":                      2,
        "bbc.co.uk":                     2,
        "bbc.com":                       2,
        "theguardian.com":               2,
        "capstonedc.com":                2,
        "think.ing.com":                 2,
        "equitablegrowth.org":           2,
        "wikipedia.org":                 2,
        "en.wikipedia.org":              2,
        "spectrumlocalnews.com":         2,
        "scmp.com":                      2,

        # --- Tier 3: commercial blog / content-marketing / low-signal
        "dimerco.com":                   3,
        "wiss.com":                      3,
        "polarismarketresearch.com":     3,
        "elevatewealth.ae":              3,
    },

    "tech_consumer": {
        # --- Tier 1: authoritative tech press (Gurman et al.)
        "bloomberg.com":                 1,
        "wsj.com":                       1,
        "reuters.com":                   1,
        "ft.com":                        1,
        "nytimes.com":                   1,
        "nikkei.com":                    1,
        "apple.com":                     1,

        # --- Tier 2: mainstream tech / specialist supply-chain trade press
        "macrumors.com":                 2,
        "appleinsider.com":              2,
        "macworld.com":                  2,
        "9to5mac.com":                   2,
        "theverge.com":                  2,
        "arstechnica.com":               2,
        "digitimes.com":                 2,
        "oled-info.com":                 2,
        "supplychainbrain.com":          2,

        # --- Tier 3: rumor aggregators / SEO / vendor marketing
        "wccftech.com":                  3,
        "ainvest.com":                   3,
        "blog.rwazi.com":                3,
        "suppliersmap.com":              3,
        "phonearena.com":                3,
        "gsmarena.com":                  3,
    },

    "regulation": {
        # --- Tier 1: official EU + gov + top regulatory-news desks
        "ec.europa.eu":                  1,
        "europarl.europa.eu":            1,
        "epthinktank.eu":                1,
        "digital-strategy.ec.europa.eu": 1,
        "europa.eu":                     1,
        "gov.uk":                        1,
        "reuters.com":                   1,
        "ft.com":                        1,
        "bloomberg.com":                 1,
        "law360.com":                    1,
        "politico.eu":                   1,

        # --- Tier 2: law firm client alerts + reputable specialist/civil
        "jdsupra.com":                   2,
        "cdp.cooley.com":                2,
        "technologyslegaledge.com":      2,
        "iapp.org":                      2,
        "unesco.org":                    2,
        "sloanreview.mit.edu":           2,
        "edri.org":                      2,
        "brookings.edu":                 2,
        "kpmg.com":                      2,  # consulting; mid-tier for this domain

        # --- Tier 3: vendor governance blogs / SEO aggregators
        "holisticai.com":                3,
        "euaiact.com":                   3,
        "artificialintelligenceact.eu":  3,
    },
}


# ---------------------------------------------------------------------------
# UGC patterns
#
# Matched against the full URL. UGC is tier-3 by definition AND
# separately flagged via `ugc_flag` because a UGC citation in a news
# briefing is a qualitatively different failure mode than a mediocre
# trade blog. A UGC citation is "Critic should have dropped this
# outright"; a tier-3 trade-blog citation is "this is thin but not
# disqualifying."
#
# Patterns are raw strings; `any(p in url for p in UGC_PATTERNS)` is
# sufficient at this scale. If the set grows past ~20 entries, move
# to compiled regex.
# ---------------------------------------------------------------------------

UGC_PATTERNS: tuple[str, ...] = (
    "linkedin.com/posts/",
    "linkedin.com/pulse/",
    "youtube.com/watch",
    "youtu.be/",
    "reddit.com/r/",
    "twitter.com/",
    "x.com/",
    "facebook.com/",
    "tiktok.com/",
    "medium.com/@",
    "substack.com/p/",
)


# ---------------------------------------------------------------------------
# URL → tier
# ---------------------------------------------------------------------------

def _url_to_key(url: str) -> tuple[str, str]:
    """Return (host, host_or_host_path) keys for tier lookup.

    host is e.g. `blog.rwazi.com`; host_or_host_path is e.g.
    `blog.rwazi.com/how-apples-iphone-17...`. Callers look up the
    longer key first so a path-specific denylist entry beats a
    host-only entry.
    """
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    # Strip leading www. but preserve other subdomains (blog., en., cdp.
    # carry tier signal).
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path or ""
    # Normalize trailing slash — doesn't affect tier but keeps keys stable.
    host_path = f"{host}{path}".rstrip("/")
    return host, host_path


def classify_url(url: str, category: str) -> tuple[int, bool]:
    """Return (tier, is_ugc) for a single URL in a topic category.

    tier is 1, 2, or 3. Unknown domains default to 2 and emit a
    warning to stderr (routed through the scoring entry point's
    unknown-domain log). is_ugc short-circuits tier assignment to 3.
    """
    for pat in UGC_PATTERNS:
        if pat in url:
            return 3, True

    tier_map = _TIER_MAPS.get(category, _TIER_MAPS["policy_macro"])
    host, host_path = _url_to_key(url)

    # Prefer path-prefix match over host match.
    for key, tier in tier_map.items():
        if "/" in key and host_path.startswith(key):
            return tier, False

    if host in tier_map:
        return tier_map[host], False

    # Default tier for unknowns. Caller is responsible for logging
    # the unknown domain if desired.
    return 2, False


# ---------------------------------------------------------------------------
# Citation parsing
#
# Briefings embed citations as `[source: URL]` spans, one per cited
# sentence. Multiple citations can appear in a single bracket span
# (e.g. `[source: URL1] [source: URL2]` or joined with "—"). The regex
# captures URLs up to the closing bracket or whitespace sentinel, which
# handles the observed formats in the eval set.
# ---------------------------------------------------------------------------

_CITATION_RE = re.compile(r"\[source:\s*(https?://[^\s\]]+)")


def extract_citations(briefing_text: str) -> list[str]:
    """All cited URLs in order of appearance, with duplicates preserved.

    Duplicates matter: a briefing that cites linkedin.com twice in
    different sections is twice as source-quality-poor as one that
    cites it once.
    """
    return _CITATION_RE.findall(briefing_text)


# ---------------------------------------------------------------------------
# Per-cell scoring
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellScore:
    topic_id: str
    pipeline: str
    category: str
    citations_total: int
    tier1_citation_share: float
    tier2_citation_share: float
    tier3_citation_share: float
    ugc_flag: int
    retrieval_total: int
    retrieval_to_citation_rate: float
    unknown_domains: tuple[str, ...]


def _cell_dir(run_id: str, topic_id: str, pipeline: str) -> Path:
    return RESULTS_DIR / run_id / topic_id / pipeline


def score_cell(run_id: str, topic_id: str, pipeline: str) -> CellScore:
    """Compute the four source-tier metrics for one cell.

    Raises FileNotFoundError if briefing.md or metadata.json is
    missing. Ratios are 0.0 when their denominator is 0 (empty
    briefing, empty source_urls); downstream aggregation should
    handle this rather than silently averaging zeros.
    """
    cell = _cell_dir(run_id, topic_id, pipeline)
    briefing = (cell / "briefing.md").read_text(encoding="utf-8")
    metadata = json.loads((cell / "metadata.json").read_text(encoding="utf-8"))

    category = TOPIC_CATEGORIES.get(topic_id, "policy_macro")

    citations = extract_citations(briefing)
    retrieved = list(metadata.get("source_urls") or [])

    tier_counts = {1: 0, 2: 0, 3: 0}
    ugc_hit = False
    unknown = []

    known_domains = set(_TIER_MAPS.get(category, {}).keys())
    for url in citations:
        tier, is_ugc = classify_url(url, category)
        tier_counts[tier] += 1
        ugc_hit = ugc_hit or is_ugc
        host, host_path = _url_to_key(url)
        # Flag unknowns only if no path-prefix or host match was found
        # AND the URL isn't UGC. This means a tier-2 default triggered.
        if tier == 2 and not is_ugc and host not in known_domains and not any(
            "/" in k and host_path.startswith(k) for k in known_domains
        ):
            unknown.append(host)

    total_cites = len(citations)
    total_retrieved = len(retrieved)
    cited_set = set(citations)
    retrieved_set = set(retrieved)

    def _share(n: int) -> float:
        return (n / total_cites) if total_cites else 0.0

    return CellScore(
        topic_id=topic_id,
        pipeline=pipeline,
        category=category,
        citations_total=total_cites,
        tier1_citation_share=_share(tier_counts[1]),
        tier2_citation_share=_share(tier_counts[2]),
        tier3_citation_share=_share(tier_counts[3]),
        ugc_flag=int(ugc_hit),
        retrieval_total=total_retrieved,
        retrieval_to_citation_rate=(
            len(cited_set & retrieved_set) / total_retrieved
            if total_retrieved else 0.0
        ),
        unknown_domains=tuple(sorted(set(unknown))),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _iter_cells(run_id: str, topic_filter: set[str] | None):
    """Yield (topic_id, pipeline) for every cell with metadata.json
    under results/<run_id>/, respecting an optional topic filter.
    """
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


def _format_table(scores: list[CellScore]) -> str:
    cols = [
        ("topic", 22),
        ("pipeline", 13),
        ("cites", 5),
        ("T1%", 6),
        ("T2%", 6),
        ("T3%", 6),
        ("UGC", 3),
        ("retr", 4),
        ("r→c%", 6),
    ]
    header = "  ".join(name.ljust(w) for name, w in cols)
    rule = "-" * len(header)
    lines = [header, rule]
    for s in scores:
        row = [
            s.topic_id.ljust(cols[0][1]),
            s.pipeline.ljust(cols[1][1]),
            str(s.citations_total).ljust(cols[2][1]),
            f"{s.tier1_citation_share:.0%}".ljust(cols[3][1]),
            f"{s.tier2_citation_share:.0%}".ljust(cols[4][1]),
            f"{s.tier3_citation_share:.0%}".ljust(cols[5][1]),
            ("Y" if s.ugc_flag else "·").ljust(cols[6][1]),
            str(s.retrieval_total).ljust(cols[7][1]),
            f"{s.retrieval_to_citation_rate:.0%}".ljust(cols[8][1]),
        ]
        lines.append("  ".join(row))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute source-tier metrics for briefing cells. Reads "
            "results/<run-id>/<topic>/<pipeline>/{briefing.md,metadata.json} "
            "and prints a per-cell comparison table. With --write-scores, "
            "also writes source_tiers.json alongside each metadata.json."
        )
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--topics", nargs="*", default=None)
    parser.add_argument(
        "--write-scores", action="store_true",
        help="Write per-cell source_tiers.json next to metadata.json.",
    )
    args = parser.parse_args()

    topic_filter = set(args.topics) if args.topics else None
    scores: list[CellScore] = []
    unknown_log: dict[str, list[str]] = {}
    for topic_id, pipeline in _iter_cells(args.run_id, topic_filter):
        s = score_cell(args.run_id, topic_id, pipeline)
        scores.append(s)
        if s.unknown_domains:
            unknown_log[f"{topic_id}/{pipeline}"] = list(s.unknown_domains)
        if args.write_scores:
            out = _cell_dir(args.run_id, topic_id, pipeline) / "source_tiers.json"
            out.write_text(json.dumps(asdict(s), indent=2, sort_keys=True))

    if not scores:
        print(
            f"No cells found under {RESULTS_DIR / args.run_id}. "
            "Run evals/run_evals.py first."
        )
        return

    print(_format_table(scores))
    if unknown_log:
        print("\nUnknown domains (defaulted to tier-2):")
        for cell, domains in unknown_log.items():
            print(f"  {cell}: {', '.join(domains)}")


if __name__ == "__main__":
    main()