"""
Filter node.

Scores each raw article 1-5 for relevance to the topic, drops anything
below FILTER_THRESHOLD. Deduplicates by URL first.

Design notes:
- Per-article scoring (one LLM call per article) rather than batch.
  Rationale: isolated decisions beat position bias, parallel via asyncio
  gives us real concurrency, and per-call spans make Day 6 eval on
  filter correctness easy to inspect.
- Dedup before scoring: saves tokens, and the first-seen article wins
  (preserves source_query attribution from whichever planner query
  surfaced it first).
- Fails loud on schema violations inside a single scoring call, but
  individual failures don't kill the whole run — one missing score is
  not a failed briefing. Compare to the Researcher, where search
  failures should bubble up.
"""

import asyncio
import json
from typing import Any

from pydantic import ValidationError

from src.llm import FILTER_MODEL, get_chat_model
from src.schemas import Article, BriefingState, ScoredArticle

from src.config import RELEVANCE_THRESHOLD

# One shared model instance per node module. Temperature 0 for deterministic
# scoring — we want the same article to get the same score across runs
# (modulo LLM nondeterminism, which is why Day 6 eval uses distributions).
_model = get_chat_model(model=FILTER_MODEL, temperature=0.0)


SCORING_PROMPT = """You are scoring a single article for its relevance to a news briefing topic.

Topic: {topic}

Article:
  Title: {title}
  URL: {url}
  Snippet: {snippet}

Score the article's relevance on a 1-5 scale:
  1 = off-topic, should be dropped
  2 = tangentially related, low value
  3 = relevant, useful context
  4 = highly relevant, direct coverage
  5 = essential, central to the topic

Respond with ONLY a JSON object, no prose, no code fences:
{{"relevance": <int 1-5>, "rationale": "<one sentence>"}}"""


def _dedupe_by_url(raw_articles: list[dict]) -> list[Article]:
    """Rehydrate raw_articles dicts -> Article models, dedup by URL, first-seen wins.

    Malformed upstream dicts are logged and skipped rather than failing the
    whole run; the Researcher's output shape is validated at construction
    time there, so this is defense-in-depth for state-serialization drift.
    """
    seen: dict[str, Article] = {}
    for d in raw_articles:
        try:
            art = Article.model_validate(d)
        except ValidationError as e:
            print(f"[filter] skipping malformed article: {e}")
            continue
        url_key = str(art.url)
        if url_key not in seen:
            seen[url_key] = art
    return list(seen.values())


async def _score_one(article: Article, topic: str) -> ScoredArticle:
    """Score a single article. Raises on schema violation or invalid JSON."""
    prompt = SCORING_PROMPT.format(
        topic=topic,
        title=article.title,
        url=str(article.url),
        snippet=article.snippet,
    )

    # ChatAnthropic's async interface. Returns an AIMessage.
    response = await _model.ainvoke([{"role": "user", "content": prompt}])
    raw = response.content.strip() if isinstance(response.content, str) else ""

    if not raw:
        raise ValueError(f"Filter got empty response for {article.url}")

    # Handle the same "model wraps JSON in prose" failure mode the Researcher
    # hits: locate the JSON object by braces rather than assuming clean output.
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(
            f"No JSON object in filter response for {article.url}: {raw[:200]!r}"
        )

    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"Filter JSON parse failed for {article.url}: {raw!r}") from e

    # Pydantic will raise on out-of-range relevance or missing rationale.
    return ScoredArticle(
        article=article,
        relevance=parsed["relevance"],
        rationale=parsed["rationale"],
    )


async def filter_node(state: BriefingState) -> dict[str, Any]:
    """
    Filter node body.

    Reads: state["raw_articles"], state["topic"]
    Writes: scored_articles (list of dicts, only those >= RELEVANCE_THRESHOLD)
    """
    raw = state["raw_articles"]
    topic = state["topic"]

    unique_articles = _dedupe_by_url(raw)

    if not unique_articles:
        return {"scored_articles": []}

    # Real parallelism. return_exceptions=True means a single scoring failure
    # doesn't cancel sibling tasks — log and continue.
    scored_results = await asyncio.gather(
        *(_score_one(art, topic) for art in unique_articles),
        return_exceptions=True,
    )

    kept: list[dict] = []
    for art, result in zip(unique_articles, scored_results):
        if isinstance(result, Exception):
            print(f"[filter] scoring failed for {art.url}: {result}")
            continue
        if result.relevance >= RELEVANCE_THRESHOLD:
            kept.append(result.model_dump(mode="json"))

    return {"scored_articles": kept}