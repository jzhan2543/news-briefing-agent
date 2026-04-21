"""
Unit tests for src/tools/tavily_cache.py.

Fully isolated: fetch_fn is a counter-backed stub, cache dir is a tmp_path.
No network, no LLM, no project-wide env state — every test that needs
env vars uses monkeypatch and removes them at teardown automatically.

The critical test is `test_passthrough_when_cache_dir_unset`: if that
ever regresses, existing production behavior (and every existing test in
the suite) breaks silently.
"""

import json
from pathlib import Path

import pytest

from src.tools.tavily_cache import (
    _normalize_query,
    cached_tavily_search,
)


class _CountingFetch:
    """Stub Tavily fetcher. Tracks call count + arguments so tests can
    assert whether the cache actually short-circuited the call."""

    def __init__(self, results: list[dict] | None = None):
        self.calls: list[str] = []
        self._results = results if results is not None else [
            {"title": "Stub result", "url": "https://example.com", "content": "x"},
        ]

    def __call__(self, query: str) -> list[dict]:
        self.calls.append(query)
        return list(self._results)


# ---------------------------------------------------------------------------
# Passthrough mode (env var unset) — this is the critical safety test
# ---------------------------------------------------------------------------

def test_passthrough_when_cache_dir_unset(monkeypatch):
    """Production default: no TAVILY_CACHE_DIR means the cache layer is
    a pure passthrough. This protects existing behavior — every call goes
    to Tavily, no disk writes, no state."""
    monkeypatch.delenv("TAVILY_CACHE_DIR", raising=False)
    monkeypatch.delenv("TAVILY_CACHE_REFRESH", raising=False)
    fetch = _CountingFetch()

    result_a = cached_tavily_search("Fed rates", fetch)
    result_b = cached_tavily_search("Fed rates", fetch)

    assert len(fetch.calls) == 2, "no cache -> both calls hit fetch"
    assert result_a == result_b


# ---------------------------------------------------------------------------
# Record/replay — the happy path
# ---------------------------------------------------------------------------

def test_miss_populates_and_hit_replays(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("TAVILY_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("TAVILY_CACHE_REFRESH", raising=False)
    fetch = _CountingFetch([{"title": "One", "url": "https://a", "content": "c"}])

    first = cached_tavily_search("fed rates", fetch)
    assert fetch.calls == ["fed rates"], "first call is a miss -> fetches"
    assert first == [{"title": "One", "url": "https://a", "content": "c"}]

    # Exactly one entry on disk. Flat layout, one JSON file.
    entries = list(tmp_path.glob("*.json"))
    assert len(entries) == 1
    stored = json.loads(entries[0].read_text())
    assert stored["query"] == "fed rates"
    assert stored["results"] == first

    second = cached_tavily_search("fed rates", fetch)
    assert fetch.calls == ["fed rates"], "second call is a hit -> no new fetch"
    assert second == first


def test_results_round_trip_structurally(monkeypatch, tmp_path: Path):
    """The cache must preserve the full shape of Tavily's response list
    — not just strip to title/url/snippet. Future consumers may read
    fields we don't currently surface."""
    monkeypatch.setenv("TAVILY_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("TAVILY_CACHE_REFRESH", raising=False)
    fat_result = [
        {
            "title": "T",
            "url": "https://u",
            "content": "c",
            "published_date": "2026-04-21",
            "score": 0.92,
            "extra_nested": {"foo": [1, 2, 3]},
        },
    ]
    fetch = _CountingFetch(fat_result)

    cached_tavily_search("anything", fetch)  # populate
    replayed = cached_tavily_search("anything", fetch)

    assert fetch.calls == ["anything"], "single fetch — second call replays"
    assert replayed == fat_result


# ---------------------------------------------------------------------------
# Normalization — equivalent queries share an entry
# ---------------------------------------------------------------------------

def test_normalize_query_unit():
    assert _normalize_query("Fed Rates") == "fed rates"
    assert _normalize_query("  fed   rates  ") == "fed rates"
    assert _normalize_query("FED\tRATES\n") == "fed rates"


def test_whitespace_and_case_variants_share_entry(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("TAVILY_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("TAVILY_CACHE_REFRESH", raising=False)
    fetch = _CountingFetch()

    cached_tavily_search("Fed Rates", fetch)
    cached_tavily_search("  fed   rates  ", fetch)
    cached_tavily_search("FED RATES", fetch)

    assert len(fetch.calls) == 1, "three equivalent queries -> one fetch"
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_distinct_queries_get_distinct_entries(monkeypatch, tmp_path: Path):
    """Sanity check against overly-aggressive normalization collapsing
    unrelated queries."""
    monkeypatch.setenv("TAVILY_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("TAVILY_CACHE_REFRESH", raising=False)
    fetch = _CountingFetch()

    cached_tavily_search("fed rates", fetch)
    cached_tavily_search("ecb rates", fetch)

    assert len(fetch.calls) == 2
    assert len(list(tmp_path.glob("*.json"))) == 2


# ---------------------------------------------------------------------------
# Refresh mode — forces overwrite
# ---------------------------------------------------------------------------

def test_refresh_mode_forces_fetch_and_overwrite(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("TAVILY_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("TAVILY_CACHE_REFRESH", raising=False)

    first_fetch = _CountingFetch([{"title": "Old", "url": "https://a", "content": "c"}])
    cached_tavily_search("fed rates", first_fetch)

    # Re-fetch with a different stubbed result; refresh mode must overwrite.
    monkeypatch.setenv("TAVILY_CACHE_REFRESH", "1")
    second_fetch = _CountingFetch([{"title": "New", "url": "https://b", "content": "c"}])
    replayed = cached_tavily_search("fed rates", second_fetch)

    assert second_fetch.calls == ["fed rates"], "refresh forces fetch"
    assert replayed[0]["title"] == "New"

    # And the on-disk entry is now the new payload.
    entries = list(tmp_path.glob("*.json"))
    assert len(entries) == 1
    stored = json.loads(entries[0].read_text())
    assert stored["results"][0]["title"] == "New"


@pytest.mark.parametrize("flag_value", ["1", "true", "TRUE", "yes", "Yes"])
def test_refresh_mode_accepts_truthy_variants(monkeypatch, tmp_path: Path, flag_value):
    monkeypatch.setenv("TAVILY_CACHE_DIR", str(tmp_path))

    fetch_initial = _CountingFetch([{"title": "old", "url": "u", "content": "c"}])
    monkeypatch.delenv("TAVILY_CACHE_REFRESH", raising=False)
    cached_tavily_search("q", fetch_initial)  # populate normally

    monkeypatch.setenv("TAVILY_CACHE_REFRESH", flag_value)
    fetch_refresh = _CountingFetch([{"title": "new", "url": "u", "content": "c"}])
    cached_tavily_search("q", fetch_refresh)

    assert fetch_refresh.calls == ["q"], f"refresh value {flag_value!r} should trigger fetch"


def test_refresh_mode_falsy_value_still_replays(monkeypatch, tmp_path: Path):
    """TAVILY_CACHE_REFRESH=0 (or empty, or anything not in the truthy
    set) leaves replay behavior intact. Guards against a common footgun
    where someone sets the var to 0 thinking it disables refresh."""
    monkeypatch.setenv("TAVILY_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("TAVILY_CACHE_REFRESH", raising=False)

    fetch_initial = _CountingFetch()
    cached_tavily_search("q", fetch_initial)

    monkeypatch.setenv("TAVILY_CACHE_REFRESH", "0")
    fetch_second = _CountingFetch()
    cached_tavily_search("q", fetch_second)

    assert fetch_second.calls == [], "refresh=0 must not force a fetch"


# ---------------------------------------------------------------------------
# Robustness — broken cache entry degrades to a miss, not a crash
# ---------------------------------------------------------------------------

def test_malformed_cache_entry_falls_back_to_fetch(monkeypatch, tmp_path: Path):
    """If someone hand-edits an entry into invalid JSON or a wrong
    schema, the run should fall through to a fetch rather than raising.
    Paranoid parsing: never die on bad cache data."""
    monkeypatch.setenv("TAVILY_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("TAVILY_CACHE_REFRESH", raising=False)

    fetch = _CountingFetch([{"title": "fresh", "url": "u", "content": "c"}])

    # Populate first, then corrupt the file.
    cached_tavily_search("q", fetch)
    [entry] = list(tmp_path.glob("*.json"))
    entry.write_text("{not json")

    result = cached_tavily_search("q", fetch)
    assert len(fetch.calls) == 2, "malformed entry -> fall through to fetch"
    assert result[0]["title"] == "fresh"

    # And the entry was rewritten with a valid payload.
    stored = json.loads(entry.read_text())
    assert stored["results"][0]["title"] == "fresh"


def test_wrong_shape_cache_entry_falls_back_to_fetch(monkeypatch, tmp_path: Path):
    """A cache entry that's valid JSON but missing the 'results' key
    (schema drift) should also degrade to a miss."""
    monkeypatch.setenv("TAVILY_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("TAVILY_CACHE_REFRESH", raising=False)

    fetch = _CountingFetch()
    cached_tavily_search("q", fetch)
    [entry] = list(tmp_path.glob("*.json"))
    entry.write_text(json.dumps({"query": "q", "some_other_key": []}))

    cached_tavily_search("q", fetch)
    assert len(fetch.calls) == 2


# ---------------------------------------------------------------------------
# Directory creation — trailing slash, nested paths
# ---------------------------------------------------------------------------

def test_cache_dir_is_created_if_missing(monkeypatch, tmp_path: Path):
    """Users should be able to point TAVILY_CACHE_DIR at a path that
    doesn't exist yet without pre-mkdir'ing it."""
    target = tmp_path / "nested" / "cache"
    assert not target.exists()

    monkeypatch.setenv("TAVILY_CACHE_DIR", str(target))
    monkeypatch.delenv("TAVILY_CACHE_REFRESH", raising=False)

    cached_tavily_search("q", _CountingFetch())
    assert target.is_dir()
    assert len(list(target.glob("*.json"))) == 1