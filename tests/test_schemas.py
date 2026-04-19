"""Tests for the Pydantic models in src/schemas.py.

These are boundary-of-validation tests: they confirm that the schemas
reject malformed inputs. If someone accidentally relaxes a constraint
during refactor, these catch it.
"""

import pytest
from pydantic import ValidationError
from datetime import UTC, datetime
from src.schemas import (
    Article,
    BriefingFailure,
    BriefingSuccess,
    ScoredArticle,
    Summary,
)


# ---------------------------------------------------------------------------
# Article
# ---------------------------------------------------------------------------


def test_article_accepts_valid_url():
    article = Article(
        url="https://example.com/page",
        title="t",
        snippet="s",
        source_query="q",
    )
    assert str(article.url).startswith("https://")


def test_article_rejects_malformed_url():
    with pytest.raises(ValidationError):
        Article(
            url="not-a-url",
            title="t",
            snippet="s",
            source_query="q",
        )


def test_article_published_date_accepts_string_or_none():
    # Day 4 decision: published_date is str | None (web dates vary in format).
    Article(
        url="https://example.com",
        title="t",
        snippet="s",
        published_date="2025-11",
        source_query="q",
    )
    Article(
        url="https://example.com",
        title="t",
        snippet="s",
        published_date=None,
        source_query="q",
    )


# ---------------------------------------------------------------------------
# ScoredArticle
# ---------------------------------------------------------------------------


def _sample_article() -> Article:
    return Article(
        url="https://example.com",
        title="t",
        snippet="s",
        source_query="q",
    )


def test_scored_article_accepts_in_range_relevance():
    for score in (1, 2, 3, 4, 5):
        ScoredArticle(article=_sample_article(), relevance=score, rationale="r")


def test_scored_article_rejects_out_of_range_relevance():
    for bad_score in (0, 6, -1, 100):
        with pytest.raises(ValidationError):
            ScoredArticle(
                article=_sample_article(),
                relevance=bad_score,
                rationale="r",
            )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def test_summary_accepts_valid_inputs():
    Summary(
        article_url="https://example.com",
        summary="A short summary.",
        key_claims=["claim one", "claim two"],
    )


def test_summary_rejects_malformed_url():
    with pytest.raises(ValidationError):
        Summary(
            article_url="not-a-url",
            summary="s",
            key_claims=["c"],
        )


# ---------------------------------------------------------------------------
# BriefingSuccess / BriefingFailure
# ---------------------------------------------------------------------------


def test_briefing_failure_rejects_unknown_reason():
    with pytest.raises(ValidationError):
        BriefingFailure(
            topic="x" * 20,
            reason="not_a_real_reason",  # type: ignore[arg-type]
            message="m",
            run_started_at=datetime.now(UTC),
            run_failed_at=datetime.now(UTC),
        )


def test_briefing_failure_accepts_known_reasons():

    for reason in (
        "invalid_topic",
        "no_search_results",
        "no_relevant_articles",
        "max_iterations",
        "schema_violation",
        "api_error",
        "unknown",
    ):
        BriefingFailure(
            topic="x" * 20,
            reason=reason,  # type: ignore[arg-type]
            message="m",
            run_started_at=datetime.now(UTC),
            run_failed_at=datetime.now(UTC),
        )