"""
Summarizer node.

For each scored_article (Filter's output), produce a Summary with a
one-paragraph summary and 2-4 key claims. Runs in parallel over all
surviving articles via asyncio.gather.

Design notes:
- Parallel pattern matches the Filter: one LLM call per article,
  gather with return_exceptions=True. Individual summarization failures
  log and skip; the briefing can still be assembled from whatever
  succeeds. If zero summaries survive, the Formatter can handle it or
  we add a guardrail node on Day 7.
- key_claims exists to give the Formatter something structured to
  work with (and gives Day 6 eval a clean target for faithfulness
  checks — each claim should be verifiable against the source snippet).
- Temperature 0.0: we want stable summaries across runs for eval
  reproducibility, not creative rephrasing.
"""

import asyncio
import json
from typing import Any

from pydantic import ValidationError

from src.llm import SUMMARIZER_MODEL, get_chat_model
from src.schemas import BriefingState, ScoredArticle, Summary


_model = get_chat_model(model=SUMMARIZER_MODEL, temperature=0.0)


SUMMARY_PROMPT = """You are summarizing one article for a news briefing.

Topic of the briefing: {topic}

Article:
  Title: {title}
  URL: {url}
  Snippet: {snippet}

Produce a one-paragraph summary (3-5 sentences) focused on what this
article contributes to understanding the topic, and 2-4 key claims —
each a single declarative sentence that can be traced back to the
article's content.

Respond with ONLY a JSON object, no prose, no code fences:
{{"summary": "<one paragraph>", "key_claims": ["<claim 1>", "<claim 2>", ...]}}"""


def _rehydrate_scored(scored_dicts: list[dict]) -> list[ScoredArticle]:
    """dicts -> ScoredArticle models. Skip malformed entries with a log line."""
    out: list[ScoredArticle] = []
    for d in scored_dicts:
        try:
            out.append(ScoredArticle.model_validate(d))
        except ValidationError as e:
            print(f"[summarizer] skipping malformed scored_article: {e}")
    return out


async def _summarize_one(scored: ScoredArticle, topic: str) -> Summary:
    """Summarize one article. Raises on invalid JSON or schema violation."""
    article = scored.article
    prompt = SUMMARY_PROMPT.format(
        topic=topic,
        title=article.title,
        url=str(article.url),
        snippet=article.snippet,
    )

    response = await _model.ainvoke([{"role": "user", "content": prompt}])
    raw = response.content.strip() if isinstance(response.content, str) else ""
    if not raw:
        raise ValueError(f"Summarizer got empty response for {article.url}")

    # Same prose-wrapping workaround as the Filter.
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(
            f"No JSON object in summarizer response for {article.url}: {raw[:200]!r}"
        )

    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Summarizer JSON parse failed for {article.url}: {raw!r}"
        ) from e

    return Summary(
        article_url=article.url,
        summary=parsed["summary"],
        key_claims=parsed["key_claims"],
    )


async def summarizer_node(state: BriefingState) -> dict[str, Any]:
    """
    Summarizer node body.

    Reads: state["scored_articles"], state["topic"]
    Writes: summaries (list of Summary dicts)
    """
    scored = _rehydrate_scored(state["scored_articles"])
    topic = state["topic"]

    if not scored:
        return {"summaries": []}

    results = await asyncio.gather(
        *(_summarize_one(s, topic) for s in scored),
        return_exceptions=True,
    )

    summaries: list[dict] = []
    for s, result in zip(scored, results):
        if isinstance(result, Exception):
            print(f"[summarizer] failed for {s.article.url}: {result}")
            continue
        summaries.append(result.model_dump(mode="json"))

    return {"summaries": summaries}