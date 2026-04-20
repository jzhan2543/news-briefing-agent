"""
Formatter node.

Takes the Summarizer's list of Summary objects and produces the final
markdown briefing as one polished document. Single LLM call — no fan-out —
because the briefing needs a coherent narrative voice across articles,
which per-article generation can't give you.

Design notes:
- Structured output via response_format so the briefing body + metadata
  come back typed rather than as free-form text. The body itself is
  markdown-formatted prose; the Pydantic wrapper just guarantees we get
  a string back plus a few structured fields that downstream (or eval)
  can use without reparsing.
- Temperature 0.2 — the Formatter is the one place where a tiny bit of
  variance helps (cleaner transitions, less robotic voice). Still low
  enough for eval reproducibility.
- Citation discipline is enforced by construction: the prompt instructs
  the model to cite every claim to one of the source URLs, and the list
  of allowed URLs is injected directly into the prompt. A Day 6 guardrail
  can post-verify that every cited URL appears in the summaries list.
"""

from typing import Any

from pydantic import BaseModel, Field

from src.llm import FORMATTER_MODEL, get_chat_model
from src.schemas import BriefingState, Summary
from pydantic import ValidationError


class _BriefingPayload(BaseModel):
    """Structured output shape for the formatted briefing."""
    headline: str = Field(description="A short, factual headline for the briefing — no sensationalism.")
    briefing_markdown: str = Field(
        description=(
            "The full briefing as markdown. Must start with an executive summary "
            "paragraph, then a section per major theme drawn from the articles. "
            "Every factual claim must be followed by a bracketed citation like "
            "[source: <url>] drawn from the provided article list."
        )
    )


_model = get_chat_model(model=FORMATTER_MODEL, temperature=0.2)
_structured_model = _model.with_structured_output(_BriefingPayload)


FORMATTER_PROMPT = """You are writing a news briefing on the following topic:

Topic: {topic}

You have been provided summaries and key claims from {n_articles} articles.
Write a briefing that synthesizes these into a coherent document — not a
list of article summaries, but a narrative that identifies themes, contrasts
where sources disagree, and gives the reader a sense of the current state
of the topic.

Requirements:
- Start with a 2-3 sentence executive summary.
- Organize the body by theme, not by article. Multiple articles may
  contribute to the same theme.
- Every factual claim must be followed by a bracketed citation using one
  of the URLs from the article list below. Format: [source: <url>]
- Do not introduce claims that are not supported by the provided summaries.
- Do not include any article that you do not cite.
- Keep the tone factual and professional. No speculation, no editorializing.

Article summaries (cite only these URLs):
{article_block}
"""


def _rehydrate_summaries(summary_dicts: list[dict]) -> list[Summary]:
    """dicts -> Summary models. Skip malformed entries with a log line."""
    out: list[Summary] = []
    for d in summary_dicts:
        try:
            out.append(Summary.model_validate(d))
        except ValidationError as e:
            print(f"[formatter] skipping malformed summary: {e}")
    return out


def _build_article_block(summaries: list[Summary]) -> str:
    """Render the summaries into a prompt-friendly block."""
    parts: list[str] = []
    for i, s in enumerate(summaries, start=1):
        claims_block = "\n".join(f"    - {c}" for c in s.key_claims)
        parts.append(
            f"[{i}] URL: {s.article_url}\n"
            f"  Summary: {s.summary}\n"
            f"  Key claims:\n{claims_block}"
        )
    return "\n\n".join(parts)


async def formatter_node(state: BriefingState) -> dict[str, Any]:
    """
    Formatter node body.

    Reads: state["summaries"], state["topic"]
    Writes: final_briefing (str — the markdown body)
    """
    summaries = _rehydrate_summaries(state["summaries"])
    topic = state["topic"]

    if not summaries:
        # No summaries to format. Return a structured empty result rather
        # than an empty string; runner.py's success path will still fire,
        # but the briefing_markdown clearly communicates the situation.
        # (A Day 7 guardrail could short-circuit to BriefingFailure earlier.)
        return {
            "final_briefing": (
                f"# {topic}\n\n"
                f"_No articles passed the relevance filter for this topic._"
            )
        }

    prompt = FORMATTER_PROMPT.format(
        topic=topic,
        n_articles=len(summaries),
        article_block=_build_article_block(summaries),
    )

    payload: _BriefingPayload = await _structured_model.ainvoke(
        [{"role": "user", "content": prompt}]
    )

    # Compose final markdown: headline as H1 + body.
    final = f"# {payload.headline}\n\n{payload.briefing_markdown}"
    return {"final_briefing": final}