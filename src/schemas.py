"""
Schemas for the news briefing pipeline.

Three groups of types live here:

1. Artifact models (Article, ScoredArticle, Summary) — Pydantic BaseModels
   that carry data between nodes. Strict validation; fail loud on malformed input.

2. BriefingState — the LangGraph state schema, as a TypedDict with per-key
   reducers. This is the container; the artifact models are its values.

3. Result models (BriefingSuccess, BriefingFailure, BriefingResult) — the
   public-facing return type of run_briefing(). Never raised; always returned.
"""

from datetime import datetime
from operator import add
from typing import Annotated, Literal, TypedDict, Union

from pydantic import BaseModel, Field, HttpUrl


# ---------------------------------------------------------------------------
# Artifact models
# ---------------------------------------------------------------------------


class Article(BaseModel):
    """One retrieved piece of web content. Day 4: snippet-based."""

    url: HttpUrl
    title: str
    snippet: str
    published_date: datetime | None = None
    source_query: str  # which planner query produced this article


class ScoredArticle(BaseModel):
    """Article with Filter's relevance assessment."""

    article: Article
    relevance: int = Field(ge=1, le=5)
    rationale: str


class Summary(BaseModel):
    """One article summarized, with traceable provenance."""

    article_url: HttpUrl
    summary: str
    key_claims: list[str]


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class BriefingState(TypedDict):
    # inputs (overwrite — set once at graph entry)
    topic: str
    run_started_at: datetime

    # planner output (append)
    search_queries: Annotated[list[str], add]

    # researcher output (append — stored as dicts for msgpack-compat;
    # rehydrate with Article.model_validate when needed inside a node)
    raw_articles: Annotated[list[dict], add]

    # filter output (append, same convention as raw_articles)
    scored_articles: Annotated[list[dict], add]

    # summarizer output (append, same convention)
    summaries: Annotated[list[dict], add]

    # formatter output (overwrite — produced once)
    final_briefing: str


# ---------------------------------------------------------------------------
# Result models (public API of run_briefing)
# ---------------------------------------------------------------------------


class BriefingSuccess(BaseModel):
    status: Literal["success"] = "success"
    topic: str
    briefing_markdown: str
    run_started_at: datetime
    run_completed_at: datetime
    thread_id: str
    source_urls: list[str]


class BriefingFailure(BaseModel):
    status: Literal["failure"] = "failure"
    topic: str
    reason: Literal[
        "invalid_topic",
        "no_search_results",
        "no_relevant_articles",
        "max_iterations",
        "schema_violation",
        "api_error",
        "unknown",
    ]
    message: str
    run_started_at: datetime
    run_failed_at: datetime
    thread_id: str | None = None


BriefingResult = Union[BriefingSuccess, BriefingFailure]