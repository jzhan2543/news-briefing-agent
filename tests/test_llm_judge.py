"""Tests for evals/llm_judge.py.

The judge itself is LLM-backed, so tests inject a stub llm_call that
returns canned responses. This covers: citation-sentence extraction,
Tavily cache indexing, aggregation math for both parts, and the
end-to-end score_cell orchestration on synthetic cell directories.

Network calls and ChatAnthropic are NOT exercised here — those are
the province of a manual `--skip-part2` smoke run against real cells
and the disk-cache. These tests cover the deterministic surfaces.
"""

import json
from pathlib import Path

import pytest

from evals import llm_judge as lj


# ---------------------------------------------------------------------------
# extract_citations_with_sentences
# ---------------------------------------------------------------------------

def test_sentence_extraction_single_citation():
    briefing = "Tariffs rose to 15% [source: https://reuters.com/x]."
    cits = lj.extract_citations_with_sentences(briefing)
    assert len(cits) == 1
    assert cits[0].url == "https://reuters.com/x"
    assert "Tariffs rose to 15%" in cits[0].sentence
    assert cits[0].sentence.endswith("]")


def test_sentence_extraction_walks_back_to_prior_period():
    briefing = (
        "First sentence without a citation. "
        "Second sentence with one [source: https://reuters.com/x]."
    )
    cits = lj.extract_citations_with_sentences(briefing)
    assert len(cits) == 1
    # The extracted sentence must be the second one, not both.
    assert "First sentence" not in cits[0].sentence
    assert "Second sentence" in cits[0].sentence


def test_sentence_extraction_preserves_duplicate_urls():
    # Same URL cited in two separate sentences → two Citation records.
    briefing = (
        "Claim A [source: https://linkedin.com/posts/x]. "
        "Claim B [source: https://linkedin.com/posts/x]."
    )
    cits = lj.extract_citations_with_sentences(briefing)
    assert len(cits) == 2
    assert cits[0].sentence != cits[1].sentence
    assert cits[0].url == cits[1].url


def test_sentence_extraction_handles_start_of_document():
    # No prior boundary — walk-back hits the start of the string.
    briefing = "Opening claim with no prior period [source: https://a.com/x]."
    cits = lj.extract_citations_with_sentences(briefing)
    assert len(cits) == 1
    assert cits[0].sentence.startswith("Opening claim")


# ---------------------------------------------------------------------------
# load_tavily_cache_index
# ---------------------------------------------------------------------------

def test_tavily_index_builds_from_cache_files(tmp_path):
    cache = tmp_path / "tavily"
    cache.mkdir()
    (cache / "a.json").write_text(json.dumps({
        "query": "q1",
        "results": [
            {"url": "https://a.com/1", "title": "T1", "content": "snippet one"},
            {"url": "https://b.com/2", "title": "T2", "content": "snippet two"},
        ],
    }))
    (cache / "b.json").write_text(json.dumps({
        "query": "q2",
        "results": [
            # Duplicate URL — first seen wins.
            {"url": "https://a.com/1", "title": "T1-alt", "content": "different"},
            {"url": "https://c.com/3", "title": "T3", "content": "snippet three"},
        ],
    }))
    idx = lj.load_tavily_cache_index(cache)
    assert set(idx.keys()) == {"https://a.com/1", "https://b.com/2", "https://c.com/3"}
    # First-seen-wins for duplicates.
    assert idx["https://a.com/1"].content == "snippet one"


def test_tavily_index_tolerates_malformed_files(tmp_path):
    cache = tmp_path / "tavily"
    cache.mkdir()
    (cache / "good.json").write_text(json.dumps({
        "query": "q", "results": [
            {"url": "https://a.com/1", "title": "T", "content": "c"}
        ],
    }))
    (cache / "bad.json").write_text("{not json")
    (cache / "wrong-shape.json").write_text(json.dumps({"results": "not a list"}))
    idx = lj.load_tavily_cache_index(cache)
    # Good file indexed; malformed ones silently skipped.
    assert "https://a.com/1" in idx
    assert len(idx) == 1


def test_tavily_index_missing_dir_returns_empty(tmp_path):
    idx = lj.load_tavily_cache_index(tmp_path / "does-not-exist")
    assert idx == {}


# ---------------------------------------------------------------------------
# JSON parsing from judge responses
# ---------------------------------------------------------------------------

def test_parse_json_extracts_object_from_prose():
    text = 'Sure, here is the verdict:\n{"verdict": "supported", "reason": "matches."}\n'
    assert lj._parse_json_from_response(text) == {
        "verdict": "supported", "reason": "matches.",
    }


def test_parse_json_handles_code_fence():
    text = '```json\n{"verdict": "partial", "reason": "r"}\n```'
    assert lj._parse_json_from_response(text) == {
        "verdict": "partial", "reason": "r",
    }


def test_parse_json_returns_empty_on_malformed():
    assert lj._parse_json_from_response("no json here") == {}
    assert lj._parse_json_from_response("{not valid json}") == {}


# ---------------------------------------------------------------------------
# judge_claim — stubbed llm_call
# ---------------------------------------------------------------------------

def test_judge_claim_unverifiable_when_snippet_missing():
    c = lj.Citation(url="https://a.com/1", sentence="Claim.")
    called = []
    def stub(system, user, model):
        called.append(1)
        return {"verdict": "supported", "reason": "r"}
    v = lj.judge_claim(c, snippet=None, model="m", llm_call=stub)
    assert v.verdict == "unverifiable"
    # No LLM call should fire for a missing snippet — short-circuit.
    assert called == []


def test_judge_claim_unverifiable_when_snippet_empty():
    c = lj.Citation(url="https://a.com/1", sentence="Claim.")
    snippet = lj.ArticleSnippet(url="https://a.com/1", title="T", content="   ")
    called = []
    def stub(system, user, model):
        called.append(1)
        return {"verdict": "supported", "reason": "r"}
    v = lj.judge_claim(c, snippet=snippet, model="m", llm_call=stub)
    assert v.verdict == "unverifiable"
    assert called == []


def test_judge_claim_returns_stub_verdict():
    c = lj.Citation(url="https://a.com/1", sentence="Tariffs rose.")
    snippet = lj.ArticleSnippet(url="https://a.com/1", title="T", content="real content")
    def stub(system, user, model):
        assert "Tariffs rose." in user
        assert "real content" in user
        return {"verdict": "supported", "reason": "match"}
    v = lj.judge_claim(c, snippet=snippet, model="m", llm_call=stub)
    assert v.verdict == "supported"
    assert v.reason == "match"


# ---------------------------------------------------------------------------
# judge_utilization — stubbed
# ---------------------------------------------------------------------------

def test_judge_utilization_empty_retrieved_returns_empty():
    def stub(system, user, model):
        pytest.fail("should not be called when retrieved is empty")
    uv = lj.judge_utilization("briefing text", retrieved=[], model="m", llm_call=stub)
    assert uv.omissions == ()


def test_judge_utilization_caps_omissions():
    # Stub returns 10 omissions — score should only surface PART2_OMISSION_CAP.
    def stub(system, user, model):
        return {
            "omissions": [
                {"summary": f"miss {i}", "source_url": f"https://a.com/{i}",
                 "importance": "high"}
                for i in range(10)
            ],
            "note": "",
        }
    retrieved = [lj.ArticleSnippet("https://a.com/1", "t", "c")]
    uv = lj.judge_utilization("b", retrieved, "m", stub)
    assert len(uv.omissions) == lj.PART2_OMISSION_CAP


def test_judge_utilization_defaults_missing_importance_to_low():
    def stub(system, user, model):
        return {"omissions": [{"summary": "x", "source_url": "https://a.com/1"}]}
    retrieved = [lj.ArticleSnippet("https://a.com/1", "t", "c")]
    uv = lj.judge_utilization("b", retrieved, "m", stub)
    assert uv.omissions[0].importance == "low"


# ---------------------------------------------------------------------------
# score_cell end-to-end with stubbed LLM
# ---------------------------------------------------------------------------

def _write_cell(tmp_path, topic_id, pipeline, briefing, source_urls):
    cell = tmp_path / "current" / topic_id / pipeline
    cell.mkdir(parents=True)
    (cell / "briefing.md").write_text(briefing, encoding="utf-8")
    (cell / "metadata.json").write_text(
        json.dumps({"source_urls": source_urls, "status": "success"})
    )


def _make_index(*triples):
    return {
        url: lj.ArticleSnippet(url=url, title=title, content=content)
        for url, title, content in triples
    }


def test_score_cell_claim_support_rate_with_partial_credit(tmp_path, monkeypatch):
    monkeypatch.setattr(lj, "RESULTS_DIR", tmp_path)
    briefing = (
        "Claim one [source: https://a.com/1]. "
        "Claim two [source: https://b.com/2]. "
        "Claim three [source: https://c.com/3]. "
        "Claim four [source: https://d.com/4]."
    )
    _write_cell(tmp_path, "t", "single_agent", briefing, [
        "https://a.com/1", "https://b.com/2", "https://c.com/3", "https://d.com/4",
    ])
    index = _make_index(
        ("https://a.com/1", "T", "content 1"),
        ("https://b.com/2", "T", "content 2"),
        ("https://c.com/3", "T", "content 3"),
        ("https://d.com/4", "T", "content 4"),
    )

    # Stub: one of each verdict, in order.
    verdicts_iter = iter(["supported", "partial", "unsupported", "unverifiable"])

    def stub(system, user, model):
        if "supported by the source article" in system:
            return {"verdict": next(verdicts_iter), "reason": "r"}
        return {"omissions": [], "note": ""}

    score = lj.score_cell(
        "current", "t", "single_agent",
        tavily_index=index, llm_call=stub,
    )
    # 1 supported + 0.5 * 1 partial = 1.5 / 4 = 0.375
    assert score.claims_total == 4
    assert score.claims_supported == 1
    assert score.claims_partial == 1
    assert score.claims_unsupported == 1
    assert score.claims_unverifiable == 1
    assert score.claim_support_rate == pytest.approx(0.375)


def test_score_cell_unverifiable_short_circuits_without_llm(tmp_path, monkeypatch):
    """URL cited but not in cache → unverifiable, no LLM call fires."""
    monkeypatch.setattr(lj, "RESULTS_DIR", tmp_path)
    briefing = "Claim [source: https://uncached.example/x]."
    _write_cell(tmp_path, "t", "single_agent", briefing, ["https://uncached.example/x"])

    def stub(system, user, model):
        # Part 2 still fires but with an empty retrieved list (URL not
        # in index) — so only the Part 2 call should happen, not Part 1.
        assert "retrieved corpus" in system or "Your job: identify" in system
        return {"omissions": [], "note": ""}

    score = lj.score_cell(
        "current", "t", "single_agent",
        tavily_index={}, llm_call=stub, skip_part2=True,
    )
    assert score.claims_unverifiable == 1
    assert score.claims_total == 1
    assert score.claim_support_rate == 0.0


def test_score_cell_utilization_score_from_high_omissions(tmp_path, monkeypatch):
    monkeypatch.setattr(lj, "RESULTS_DIR", tmp_path)
    briefing = "Something [source: https://a.com/1]."
    _write_cell(tmp_path, "eu", "multi_agent", briefing,
                ["https://a.com/1", "https://edri.org/x"])
    index = _make_index(
        ("https://a.com/1", "T1", "content"),
        ("https://edri.org/x", "EDRi Letter", "civil society on implementation delays"),
    )

    def stub(system, user, model):
        if "supported by the source article" in system:
            return {"verdict": "supported", "reason": "r"}
        # Two high-importance omissions — EDRi civil-society framing shape.
        return {
            "omissions": [
                {"summary": "civil society warning on delay", "source_url": "https://edri.org/x",
                 "importance": "high"},
                {"summary": "omnibus centralization controversy", "source_url": "https://edri.org/x",
                 "importance": "high"},
            ],
            "note": "briefing misses the civil-society perspective",
        }

    score = lj.score_cell(
        "current", "eu", "multi_agent",
        tavily_index=index, llm_call=stub,
    )
    assert score.omissions_high == 2
    # 1 - (2/5) = 0.6
    assert score.retrieval_utilization_score == pytest.approx(0.6)


def test_score_cell_skip_part2_does_not_call_part2(tmp_path, monkeypatch):
    monkeypatch.setattr(lj, "RESULTS_DIR", tmp_path)
    briefing = "Claim [source: https://a.com/1]."
    _write_cell(tmp_path, "t", "single_agent", briefing, ["https://a.com/1"])
    index = _make_index(("https://a.com/1", "T", "content"))

    def stub(system, user, model):
        # Part 2 system prompt contains "retrieved corpus"; fail if that fires.
        assert "retrieved corpus" not in system, "Part 2 should be skipped"
        return {"verdict": "supported", "reason": "r"}

    score = lj.score_cell(
        "current", "t", "single_agent",
        tavily_index=index, llm_call=stub, skip_part2=True,
    )
    assert score.claim_support_rate == 1.0
    # Skipped Part 2 leaves the score at its default 1.0 and omissions at 0.
    assert score.retrieval_utilization_score == 1.0
    assert score.omissions_high == 0


def test_score_cell_empty_briefing(tmp_path, monkeypatch):
    monkeypatch.setattr(lj, "RESULTS_DIR", tmp_path)
    _write_cell(tmp_path, "t", "single_agent",
                briefing="No citations at all.", source_urls=[])

    def stub(system, user, model):
        pytest.fail("should not call LLM on zero-citation, zero-retrieval briefing")

    score = lj.score_cell(
        "current", "t", "single_agent",
        tavily_index={}, llm_call=stub,
    )
    assert score.claims_total == 0
    assert score.claim_support_rate == 0.0