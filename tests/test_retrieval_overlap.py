"""Tests for the retrieval_overlap module."""

import json
from pathlib import Path

import pytest

from evals import retrieval_overlap as ro


# ---------------------------------------------------------------------------
# URL normalization
# ---------------------------------------------------------------------------


class TestNormalizeUrl:
    def test_strips_www(self):
        assert ro._normalize_url("https://www.reuters.com/foo") == ro._normalize_url(
            "https://reuters.com/foo"
        )

    def test_strips_trailing_slash(self):
        assert ro._normalize_url("https://reuters.com/foo/") == ro._normalize_url(
            "https://reuters.com/foo"
        )

    def test_preserves_non_www_subdomain(self):
        # Subdomains other than www. carry signal (blog., en., etc.)
        assert ro._normalize_url("https://blog.reuters.com/foo") != ro._normalize_url(
            "https://reuters.com/foo"
        )

    def test_lowercases_host(self):
        assert ro._normalize_url("https://REUTERS.com/foo") == ro._normalize_url(
            "https://reuters.com/foo"
        )


# ---------------------------------------------------------------------------
# compute_overlap
# ---------------------------------------------------------------------------


class TestComputeOverlap:
    def test_identical_sets_are_one(self):
        urls = ["https://reuters.com/a", "https://ft.com/b"]
        jaccard, _, only_single, only_multi = ro.compute_overlap(urls, urls)
        assert jaccard == 1.0
        assert only_single == set()
        assert only_multi == set()

    def test_disjoint_sets_are_zero(self):
        jaccard, _, only_single, only_multi = ro.compute_overlap(
            ["https://reuters.com/a"], ["https://ft.com/b"]
        )
        assert jaccard == 0.0
        assert len(only_single) == 1
        assert len(only_multi) == 1

    def test_partial_overlap(self):
        # 2 shared, 1 only-single, 1 only-multi → intersection=2, union=4 → 0.5
        single = ["https://a.com", "https://b.com", "https://c.com"]
        multi = ["https://a.com", "https://b.com", "https://d.com"]
        jaccard, _, only_single, only_multi = ro.compute_overlap(single, multi)
        assert jaccard == 0.5
        assert len(only_single) == 1
        assert len(only_multi) == 1

    def test_both_empty_is_vacuously_identical(self):
        # Failed cells that write no URLs shouldn't raise division-by-zero.
        # 1.0 is the defensible value — they didn't disagree on anything.
        jaccard, _, only_single, only_multi = ro.compute_overlap([], [])
        assert jaccard == 1.0
        assert only_single == set()
        assert only_multi == set()

    def test_one_empty_is_zero(self):
        jaccard, _, only_single, only_multi = ro.compute_overlap([], ["https://a.com"])
        assert jaccard == 0.0
        assert only_single == set()
        assert len(only_multi) == 1

    def test_normalization_applied(self):
        # www. and trailing-slash differences should NOT reduce overlap —
        # they're the same source.
        single = ["https://www.reuters.com/foo/"]
        multi = ["https://reuters.com/foo"]
        jaccard, _, _, _ = ro.compute_overlap(single, multi)
        assert jaccard == 1.0

    def test_duplicates_within_one_side_dont_inflate(self):
        # Sets, not multisets — same URL twice in single shouldn't count twice.
        single = ["https://a.com", "https://a.com", "https://b.com"]
        multi = ["https://a.com", "https://b.com"]
        jaccard, _, _, _ = ro.compute_overlap(single, multi)
        assert jaccard == 1.0


# ---------------------------------------------------------------------------
# score_topic — file-reading integration
# ---------------------------------------------------------------------------


def _write_metadata(path: Path, source_urls: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"source_urls": source_urls, "status": "success"}))


class TestScoreTopic:
    def test_reads_both_pipelines_and_computes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ro, "RESULTS_DIR", tmp_path)

        topic_dir = tmp_path / "runA" / "mytopic"
        _write_metadata(
            topic_dir / "single_agent" / "metadata.json",
            ["https://a.com", "https://b.com", "https://c.com"],
        )
        _write_metadata(
            topic_dir / "multi_agent" / "metadata.json",
            ["https://a.com", "https://b.com"],
        )

        result = ro.score_topic("runA", "mytopic")
        assert result.topic_id == "mytopic"
        assert result.single_agent_count == 3
        assert result.multi_agent_count == 2
        assert result.intersection_count == 2
        assert result.union_count == 3
        assert abs(result.jaccard - (2 / 3)) < 1e-9
        # 2/3 ≈ 0.667 < 0.7
        assert result.passes_threshold is False
        # The "c" URL is the asymmetry; the exact normalized form
        # depends on _url_to_key and isn't what this test is about.
        assert len(result.only_single_agent) == 1
        assert "c.com" in result.only_single_agent[0]
        assert result.only_multi_agent == []

    def test_passes_threshold_flag(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ro, "RESULTS_DIR", tmp_path)

        topic_dir = tmp_path / "runA" / "mytopic"
        urls = [f"https://s{i}.com" for i in range(10)]
        _write_metadata(topic_dir / "single_agent" / "metadata.json", urls)
        _write_metadata(topic_dir / "multi_agent" / "metadata.json", urls)

        result = ro.score_topic("runA", "mytopic")
        assert result.jaccard == 1.0
        assert result.passes_threshold is True

    def test_custom_threshold_changes_passing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ro, "RESULTS_DIR", tmp_path)

        topic_dir = tmp_path / "runA" / "mytopic"
        # jaccard will be 0.75 (3 shared / 4 union)
        _write_metadata(
            topic_dir / "single_agent" / "metadata.json",
            ["https://a.com", "https://b.com", "https://c.com", "https://d.com"],
        )
        _write_metadata(
            topic_dir / "multi_agent" / "metadata.json",
            ["https://a.com", "https://b.com", "https://c.com"],
        )

        result_strict = ro.score_topic("runA", "mytopic", threshold=0.8)
        assert result_strict.passes_threshold is False

        result_lenient = ro.score_topic("runA", "mytopic", threshold=0.5)
        assert result_lenient.passes_threshold is True

    def test_missing_metadata_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ro, "RESULTS_DIR", tmp_path)

        topic_dir = tmp_path / "runA" / "mytopic"
        _write_metadata(
            topic_dir / "single_agent" / "metadata.json", ["https://a.com"]
        )
        # multi_agent metadata never written

        with pytest.raises(FileNotFoundError):
            ro.score_topic("runA", "mytopic")