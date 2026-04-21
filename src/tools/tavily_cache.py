"""
Disk-backed record/replay cache for Tavily search responses.

Why this exists: pipeline comparisons (single-agent vs. multi-agent on the
same topic) are confounded when Tavily returns different article sets
across runs of the same query. This cache pins the article set across
runs so evaluation measures architecture, not retrieval variance.

Activation: off by default. Set TAVILY_CACHE_DIR to an existing (or
creatable) directory to enable. When enabled:
- Cache miss -> call Tavily, write the raw results to disk, return them.
- Cache hit  -> read and return; no Tavily call, no network.

Set TAVILY_CACHE_REFRESH=1 to force a Tavily call and rewrite on every
lookup (useful for periodic regeneration of the whole cache).

Storage layout: flat directory of <hash>.json files. Each file stores
the original (unnormalized) query alongside the results, so `ls` + `cat`
is enough to inspect what's pinned. No TTL — staleness is the feature.

Key normalization: lowercase + whitespace collapse. Equivalent queries
("  Fed rates ", "fed  rates") share one cache entry. `max_results` is
not part of the key because the call site hard-codes it to 5; if that
assumption changes, elevate it into the key.
"""

import hashlib
import json
import os
import re
from pathlib import Path


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_query(query: str) -> str:
    """Lowercase + collapse whitespace. Keeps semantically-equivalent
    queries sharing a single cache entry."""
    return _WHITESPACE_RE.sub(" ", query.strip().lower())


def _cache_dir() -> Path | None:
    """Returns the cache directory if caching is enabled, else None.

    Called on every cached_tavily_search invocation so the env var can be
    flipped mid-process (useful for tests that want to toggle behavior
    without re-importing the module).
    """
    raw = os.environ.get("TAVILY_CACHE_DIR")
    if not raw:
        return None
    path = Path(raw).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _refresh_mode() -> bool:
    """Returns True when TAVILY_CACHE_REFRESH is set to a truthy value.

    In refresh mode the cache is write-only on this call — we skip the
    read, hit Tavily, and overwrite the entry. Subsequent calls without
    the flag replay the new entry."""
    return os.environ.get("TAVILY_CACHE_REFRESH", "").lower() in {"1", "true", "yes"}


def _cache_key(normalized_query: str) -> str:
    """SHA-1 of the normalized query. SHA-1 is fine here — this is a
    filename derivation, not a security boundary."""
    return hashlib.sha1(normalized_query.encode("utf-8")).hexdigest()


def _entry_path(cache_dir: Path, normalized_query: str) -> Path:
    return cache_dir / f"{_cache_key(normalized_query)}.json"


def _read_entry(path: Path) -> list[dict] | None:
    """Read a cache entry. Returns None if the file is missing or
    malformed — caller treats that as a miss rather than raising, because
    a broken cache entry shouldn't break a run."""
    try:
        raw = path.read_text()
    except OSError:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    results = payload.get("results")
    if not isinstance(results, list):
        return None
    return results


def _write_entry(path: Path, query: str, results: list[dict]) -> None:
    """Write a cache entry. Stores the original (pre-normalization) query
    for debugging; the filename hash is keyed on the normalized form."""
    payload = {"query": query, "results": results}
    path.write_text(json.dumps(payload, indent=2))


def cached_tavily_search(
    query: str,
    fetch_fn,
) -> list[dict]:
    """Record/replay wrapper around a Tavily fetch function.

    When TAVILY_CACHE_DIR is unset, this is a thin passthrough to fetch_fn
    — the project's behavior outside of eval contexts is unchanged.

    fetch_fn is injected rather than imported so the cache module stays
    testable without patching network code. Expected signature:
    `fetch_fn(query: str) -> list[dict]`.
    """
    cache_dir = _cache_dir()
    if cache_dir is None:
        # Caching disabled — behave exactly like the uncached call site.
        return fetch_fn(query)

    normalized = _normalize_query(query)
    path = _entry_path(cache_dir, normalized)

    if not _refresh_mode():
        cached = _read_entry(path)
        if cached is not None:
            return cached

    # Miss (or refresh): call through and record.
    results = fetch_fn(query)
    _write_entry(path, query, results)
    return results