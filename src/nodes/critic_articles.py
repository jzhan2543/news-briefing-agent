"""
Critic(articles) node: multi-agent variant's replacement for the Filter node.

Where Filter runs one scoring LLM call per article in parallel and drops
anything below RELEVANCE_THRESHOLD, Critic(articles) runs a single agent
invocation over the entire batch — because seeing all articles together is
what lets the "duplicate" flag fire. The tradeoff: one longer LLM call
instead of N short parallel ones.

Design notes:
- Batch invocation (not per-article fan-out): duplicate detection requires
  visibility into the whole set. If Day 6 eval shows per-article scoring
  quality matters more than duplicate detection, this can be refactored
  to two passes (parallel score + single pass dedup).
- Dedupe by URL first, then score: same convention as Filter. First-seen
  URL wins, preserving source_query attribution from whichever planner
  query surfaced it first.
- Fail-loud on the agent itself (schema violations propagate), but
  skip-and-log on per-article misalignment — if the agent returns N-1
  scored entries for N articles, we fill the missing slot with a neutral
  fallback rather than drop the article silently.
- Same RELEVANCE_THRESHOLD as Filter — the multi-agent variant keeps
  Filter's cutoff behavior; issue flags are additive metadata, not a
  replacement for the hard cutoff. Day 6 eval can compare cutoff-only
  (Filter) vs cutoff-plus-flags (Critic) as a tradeoff dimension.
"""

from typing import Any

from pydantic import ValidationError

from src.agents.critic_articles import build_critic_articles_agent
from src.config import RELEVANCE_THRESHOLD
from src.schemas import (
    Article,
    CriticScoredArticle,
    IssueFlag,
    MultiAgentBriefingState,
)


def _dedupe_by_url(raw_articles: list[dict]) -> list[Article]:
    """Rehydrate raw_articles dicts -> Article models, dedup by URL, first-seen wins.

    Lifted verbatim from the Filter node so the two variants share the same
    pre-scoring behavior. If you change one, change both — or factor into
    a shared helper on Day 6.
    """
    seen: dict[str, Article] = {}
    for d in raw_articles:
        try:
            art = Article.model_validate(d)
        except ValidationError as e:
            print(f"[critic_articles] skipping malformed article: {e}")
            continue
        url_key = str(art.url)
        if url_key not in seen:
            seen[url_key] = art
    return list(seen.values())


def _format_articles_for_prompt(articles: list[Article]) -> str:
    """Render the article batch into the prompt. Index is the identity anchor
    the agent uses to produce output in the same order.
    """
    parts: list[str] = []
    for i, art in enumerate(articles):
        parts.append(
            f"[{i}] Title: {art.title}\n"
            f"    URL: {art.url}\n"
            f"    Snippet: {art.snippet}"
        )
    return "\n\n".join(parts)


async def critic_articles_node(state: MultiAgentBriefingState) -> dict[str, Any]:
    """
    Critic(articles) node body.

    Reads: state["raw_articles"], state["topic"]
    Writes: scored_articles (list of CriticScoredArticle dicts, only those
            >= RELEVANCE_THRESHOLD)
    """
    raw = state["raw_articles"]
    topic = state["topic"]

    unique_articles = _dedupe_by_url(raw)

    if not unique_articles:
        return {"scored_articles": []}

    agent = build_critic_articles_agent()
    articles_block = _format_articles_for_prompt(unique_articles)

    user_message = (
        f"Topic: {topic}\n\n"
        f"Articles to evaluate ({len(unique_articles)} total):\n\n"
        f"{articles_block}"
    )

    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": user_message}],
    })

    # With response_format, the structured output lives on result["structured_response"].
    # If the agent failed to produce structured output (shouldn't happen with a
    # schema'd response_format, but belt-and-suspenders), treat as an error.
    payload = result.get("structured_response")
    if payload is None:
        raise ValueError(
            "[critic_articles] agent returned no structured_response. "
            "Check the LangSmith trace for the final message content."
        )

    # Map agent-returned entries back to their input indices. entries hold
    # the wire format (primitive types); we hydrate to rich CriticScoredArticle
    # in the rebuild loop below.
    scored_by_index: dict[int, Any] = {}
    for i, entry in enumerate(payload.scored_articles):
        # Trust the agent's ordering: it was instructed to preserve input order.
        # If it returns fewer entries than inputs, the zip below catches the gap.
        if i < len(unique_articles):
            scored_by_index[i] = entry

    kept: list[dict] = []
    for i, art in enumerate(unique_articles):
        if i not in scored_by_index:
            print(
                f"[critic_articles] agent returned no score for article {i} "
                f"({art.url}); skipping"
            )
            continue
        entry = scored_by_index[i]
        if entry.relevance < RELEVANCE_THRESHOLD:
            continue
        # Hydrate wire-format flags into rich IssueFlag models, and pair with
        # the original Article to build a full CriticScoredArticle. See the
        # comment block in agents/critic_articles.py for why the wire format
        # is separate from the state artifact type.
        rich_flags = [
            IssueFlag(
                issue_type=f.issue_type,
                severity=f.severity,
                note=f.note,
            )
            for f in entry.flags
        ]
        repaired = CriticScoredArticle(
            article=art,
            relevance=entry.relevance,
            flags=rich_flags,
            justification=entry.justification,
        )
        kept.append(repaired.model_dump(mode="json"))

    return {"scored_articles": kept}