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
    published_date: str | None = None       # was: datetime | None = None
    source_query: str


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
# Critic (multi-agent variant) artifact models
# ---------------------------------------------------------------------------


IssueType = Literal[
    "off_topic",
    "low_credibility",
    "unsupported_claim",
    "outdated",
    "duplicate",
]


class IssueFlag(BaseModel):
    """One structured concern the Critic raises about an article.

    Closed enum on issue_type by design: letting the model invent categories
    at runtime would be useful for exploration but terrible for downstream
    consumption — the Writer and Critic(draft) both read these flags and
    need to know what they can see.
    """

    issue_type: IssueType
    severity: Literal["low", "medium", "high"]
    note: str = Field(..., min_length=3, description="One sentence explaining the flag.")


class CriticScoredArticle(BaseModel):
    """Critic(articles)'s richer alternative to Filter's ScoredArticle.

    Carries the same relevance score (1-5) but adds structured issue flags
    and a justification. Named distinctly from ScoredArticle to make it
    unambiguous which pipeline produced the scored_articles entry when
    you're debugging a trace or eval output.
    """

    article: Article
    relevance: int = Field(ge=1, le=5)
    flags: list[IssueFlag] = Field(default_factory=list)
    justification: str = Field(..., min_length=3)


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
    

class MultiAgentBriefingState(BriefingState):
    """BriefingState extended with the multi-agent revision-loop fields.

    Inherits all existing fields (topic, run_started_at, search_queries,
    raw_articles, scored_articles, summaries, final_briefing) with their
    existing reducers intact. The four added fields are overwrite-on-update
    (no reducer annotation) because:

    - revision_count: counter; each write is a full new value.
    - critic_verdict: latest verdict wins.
    - critic_feedback: each revision round replaces the previous feedback.
    - draft: each Writer invocation produces a full new draft.

    Revision history is NOT accumulated in state. What's "remembered"
    across revision rounds is implicit in the Critic's feedback (which
    the Writer sees next round). Adding draft_history would bloat state
    and context with no downstream benefit.
    """

    revision_count: int
    critic_verdict: Literal["accept", "revise"] | None
    critic_feedback: str | None
    draft: str | None


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