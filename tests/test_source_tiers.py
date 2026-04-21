"""Tests for evals/source_tiers.py.

Covers: URL→tier classification per category, UGC short-circuit,
path-prefix vs host-only key precedence, citation regex parsing,
and end-to-end scoring on a synthetic cell directory.
"""

import json
from pathlib import Path

import pytest

from evals import source_tiers as st


# ---------------------------------------------------------------------------
# classify_url: tier assignment
# ---------------------------------------------------------------------------

def test_tier1_policy_macro():
    tier, ugc = st.classify_url("https://www.reuters.com/article/x", "policy_macro")
    assert tier == 1
    assert ugc is False


def test_tier3_policy_macro_commercial_blog():
    tier, ugc = st.classify_url(
        "https://wiss.com/trump-tariffs-2026-financial-impact-on-us-manufacturers/",
        "policy_macro",
    )
    assert tier == 3


def test_tier_is_category_conditional_bloomberg_policy_vs_tech():
    # Bloomberg is tier-1 in both; chosen because it's the unambiguous
    # shared reputable source. The real point of this test is that
    # the tier map is *looked up per category* — exercise the code
    # path even when the result happens to match.
    tier_p, _ = st.classify_url("https://www.bloomberg.com/x", "policy_macro")
    tier_t, _ = st.classify_url("https://www.bloomberg.com/x", "tech_consumer")
    assert tier_p == tier_t == 1


def test_tier2_vs_tier3_split_matches_tech_expectations():
    # MacRumors/AppleInsider are tier-2 for tech (specialist but
    # rumor-heavy); WCCFTech is tier-3 (aggregator).
    t_macrumors, _ = st.classify_url(
        "https://www.macrumors.com/roundup/iphone-17/", "tech_consumer"
    )
    t_wccf, _ = st.classify_url(
        "https://wccftech.com/apple-slashes-iphone-17-air-production/",
        "tech_consumer",
    )
    assert t_macrumors == 2
    assert t_wccf == 3


def test_subdomain_preserved_for_tier_signal():
    # blog.rwazi.com is tier-3 (marketing blog); rwazi.com proper isn't
    # in the map — the subdomain carries the signal.
    tier, _ = st.classify_url(
        "https://blog.rwazi.com/how-apples-iphone-17-is-redefining/",
        "tech_consumer",
    )
    assert tier == 3


def test_unknown_domain_defaults_to_tier2():
    # Unknowns get tier-2 (neither reward nor penalty) and are logged
    # by the caller, not by classify_url itself.
    tier, ugc = st.classify_url("https://random-unknown-site.example/x", "policy_macro")
    assert tier == 2
    assert ugc is False


# ---------------------------------------------------------------------------
# classify_url: UGC short-circuit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "https://www.linkedin.com/posts/sunish-bhatia-6280a124a_apple-activity-7382/",
    "https://www.youtube.com/watch?v=SNRzCTl3siU",
    "https://youtu.be/abc123",
    "https://www.reddit.com/r/apple/comments/xxx/",
    "https://twitter.com/user/status/123",
    "https://x.com/user/status/123",
    "https://medium.com/@personal/my-take-on-ai-act",
    "https://someone.substack.com/p/iphone-rumors",
])
def test_ugc_patterns_flagged(url):
    tier, ugc = st.classify_url(url, "tech_consumer")
    assert ugc is True
    assert tier == 3


def test_ugc_flag_overrides_category():
    # A LinkedIn post is UGC regardless of topic category.
    _, ugc_tech = st.classify_url(
        "https://www.linkedin.com/posts/someone-abc/", "tech_consumer"
    )
    _, ugc_policy = st.classify_url(
        "https://www.linkedin.com/posts/someone-abc/", "policy_macro"
    )
    assert ugc_tech is True
    assert ugc_policy is True


# ---------------------------------------------------------------------------
# extract_citations
# ---------------------------------------------------------------------------

def test_extract_single_citation():
    text = "Tariffs rose [source: https://reuters.com/x]."
    assert st.extract_citations(text) == ["https://reuters.com/x"]


def test_extract_preserves_duplicates():
    # A briefing that cites the same URL twice in different paragraphs
    # is *twice as source-influenced* by it. Duplicates must be kept.
    text = (
        "Claim A [source: https://linkedin.com/posts/x]. "
        "Claim B [source: https://linkedin.com/posts/x]."
    )
    assert st.extract_citations(text) == [
        "https://linkedin.com/posts/x",
        "https://linkedin.com/posts/x",
    ]


def test_extract_handles_multiple_citations_one_sentence():
    text = (
        "Fact [source: https://a.com/1] [source: https://b.com/2]."
    )
    assert st.extract_citations(text) == ["https://a.com/1", "https://b.com/2"]


def test_extract_tolerates_whitespace_after_source_marker():
    # Observed format allows `[source: URL]` with variable whitespace.
    text = "X [source:    https://reuters.com/y]."
    assert st.extract_citations(text) == ["https://reuters.com/y"]


def test_extract_ignores_non_http_tokens():
    # "[source: not-a-url]" shouldn't be captured — the regex requires
    # an http(s) scheme so malformed citations are surfaced as absent,
    # not silently accepted.
    text = "X [source: not-a-url]."
    assert st.extract_citations(text) == []


# ---------------------------------------------------------------------------
# score_cell — end-to-end on a synthetic cell directory
# ---------------------------------------------------------------------------

def _write_cell(
    tmp_path: Path, topic_id: str, pipeline: str,
    briefing: str, source_urls: list[str],
) -> None:
    cell = tmp_path / "current" / topic_id / pipeline
    cell.mkdir(parents=True)
    (cell / "briefing.md").write_text(briefing, encoding="utf-8")
    (cell / "metadata.json").write_text(
        json.dumps({"source_urls": source_urls, "status": "success"}),
        encoding="utf-8",
    )


def test_score_cell_on_iphone_style_fixture(tmp_path, monkeypatch):
    """Single-agent iPhone shape: cites LinkedIn and YouTube, low tier-1."""
    monkeypatch.setattr(st, "RESULTS_DIR", tmp_path)
    briefing = (
        "Apple shifts production [source: https://www.digitimes.com/news/x]. "
        "Per LinkedIn [source: https://www.linkedin.com/posts/sunish-bhatia/]. "
        "YouTube analysis [source: https://www.youtube.com/watch?v=abc]. "
        "Aggregator says [source: https://wccftech.com/y]."
    )
    _write_cell(
        tmp_path, "iphone_launch_rumors", "single_agent",
        briefing,
        source_urls=[
            "https://www.digitimes.com/news/x",
            "https://www.linkedin.com/posts/sunish-bhatia/",
            "https://www.youtube.com/watch?v=abc",
            "https://wccftech.com/y",
            "https://www.macrumors.com/roundup/iphone-17/",
        ],
    )
    score = st.score_cell("current", "iphone_launch_rumors", "single_agent")
    assert score.citations_total == 4
    assert score.ugc_flag == 1
    # LinkedIn + YouTube + WCCFTech = 3 of 4 tier-3, DigiTimes = 1 tier-2.
    assert score.tier3_citation_share == pytest.approx(3 / 4)
    assert score.tier2_citation_share == pytest.approx(1 / 4)
    assert score.tier1_citation_share == 0.0
    # 4 of 5 retrieved URLs are cited; macrumors retrieved but not cited.
    assert score.retrieval_to_citation_rate == pytest.approx(4 / 5)


def test_score_cell_tariffs_style_fixture(tmp_path, monkeypatch):
    """Multi-agent tariffs shape: Critic dropped one low-tier source."""
    monkeypatch.setattr(st, "RESULTS_DIR", tmp_path)
    briefing = (
        "Policy shift [source: https://www.reuters.com/x]. "
        "JPM estimate [source: https://www.jpmorgan.com/insights/y]. "
        "Commercial blog [source: https://wiss.com/trump-tariffs-2026/]."
    )
    _write_cell(
        tmp_path, "tariffs_2026", "multi_agent",
        briefing,
        source_urls=[
            "https://www.reuters.com/x",
            "https://www.jpmorgan.com/insights/y",
            "https://wiss.com/trump-tariffs-2026/",
            # Retrieved but not cited — simulates Critic drop.
            "https://www.polarismarketresearch.com/blog/x",
        ],
    )
    score = st.score_cell("current", "tariffs_2026", "multi_agent")
    assert score.citations_total == 3
    assert score.ugc_flag == 0
    assert score.tier1_citation_share == pytest.approx(2 / 3)
    assert score.tier3_citation_share == pytest.approx(1 / 3)
    # 3 of 4 retrieved cited; Polaris dropped.
    assert score.retrieval_to_citation_rate == pytest.approx(3 / 4)


def test_score_cell_empty_briefing_returns_zero_ratios(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "RESULTS_DIR", tmp_path)
    _write_cell(
        tmp_path, "tariffs_2026", "single_agent",
        briefing="No citations here at all.",
        source_urls=["https://www.reuters.com/x"],
    )
    score = st.score_cell("current", "tariffs_2026", "single_agent")
    assert score.citations_total == 0
    assert score.tier1_citation_share == 0.0
    assert score.tier3_citation_share == 0.0
    assert score.ugc_flag == 0
    # Retrieval denominator is 1, no citations ∩ retrieved = 0/1.
    assert score.retrieval_to_citation_rate == 0.0


def test_score_cell_unknown_domain_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "RESULTS_DIR", tmp_path)
    briefing = "Random claim [source: https://some-unknown-tracker.example/y]."
    _write_cell(
        tmp_path, "tariffs_2026", "single_agent",
        briefing,
        source_urls=["https://some-unknown-tracker.example/y"],
    )
    score = st.score_cell("current", "tariffs_2026", "single_agent")
    # Unknown defaults to tier-2, is NOT flagged as UGC, and surfaces
    # in unknown_domains.
    assert score.tier2_citation_share == 1.0
    assert score.ugc_flag == 0
    assert "some-unknown-tracker.example" in score.unknown_domains