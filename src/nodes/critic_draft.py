"""
Critic(draft) node: review the Writer's current draft, emit a verdict.

Always writes final_briefing = draft, regardless of verdict. The conditional
edge downstream decides whether to loop back to the Writer or go to END;
making Critic(draft) always write final_briefing keeps the conditional
edge as pure routing and guarantees that force-accept at revision_count=2
produces a valid final_briefing without the edge having to touch state.

Design notes:
- Defensive fallback on malformed verdict. With response_format's Literal
  enum this shouldn't fire, but if the agent somehow produces a non-enum
  value (tool-call weirdness, truncation, API error), default to accept
  and log. Don't loop on a broken Critic — the revision cap is a safety
  bound, not a hiding place for bugs.
- scored_articles rehydration is best-effort. If every entry fails to
  validate as CriticScoredArticle, we still run the Critic on whatever
  the draft claims; the faithfulness check degrades but doesn't hard-fail.
  Evaluation on Day 6 should catch the signal ("critic runs with empty
  sources" is a failure mode we want visible).
- Pretty-prints articles for the prompt rather than dumping JSON. The
  Critic has to reason about claims and citations; a readable format is
  tokens well spent.
"""

from typing import Any

from pydantic import ValidationError

from src.agents.critic_draft import build_critic_draft_agent
from src.schemas import CriticScoredArticle, MultiAgentBriefingState


def _rehydrate_scored(scored_dicts: list[dict]) -> list[CriticScoredArticle]:
    """dicts -> CriticScoredArticle models. Skip malformed entries with a log line.

    Same convention as Filter/Summarizer's rehydrate helpers. Failures here
    degrade Critic quality (fewer sources to check faithfulness against) but
    don't hard-fail the node — we want the Critic to still produce a verdict
    so the graph can terminate cleanly.
    """
    out: list[CriticScoredArticle] = []
    for d in scored_dicts:
        try:
            out.append(CriticScoredArticle.model_validate(d))
        except ValidationError as e:
            print(f"[critic_draft] skipping malformed scored_article: {e}")
    return out


def _format_sources_for_prompt(scored: list[CriticScoredArticle]) -> str:
    """Render the scored articles as a readable block for the Critic.

    Includes flags because flag-resolution is priority 3 in the Critic's
    rubric — the Critic has to see which articles were pre-flagged to
    apply the caveating rule.
    """
    parts: list[str] = []
    for i, s in enumerate(scored, start=1):
        flag_str = (
            ", ".join(f"{f.issue_type}({f.severity})" for f in s.flags)
            if s.flags
            else "none"
        )
        parts.append(
            f"[{i}] URL: {s.article.url}\n"
            f"    Title: {s.article.title}\n"
            f"    Snippet: {s.article.snippet}\n"
            f"    Relevance: {s.relevance}/5\n"
            f"    Flags: {flag_str}"
        )
    return "\n\n".join(parts)


async def critic_draft_node(state: MultiAgentBriefingState) -> dict[str, Any]:
    """
    Critic(draft) node body.

    Reads: state["draft"], state["scored_articles"], state["topic"]
    Writes: critic_verdict, critic_feedback, final_briefing (always = draft)

    Fallbacks:
      - Missing draft: return verdict="accept" with empty feedback and
        empty final_briefing. This shouldn't happen (the graph topology
        only reaches Critic(draft) after the Writer runs), but if it does
        we want to terminate rather than loop on nothing.
      - Malformed agent output: fall back to accept with a log line. The
        revision cap is a safety bound, not a hiding place for bugs.
    """
    draft = state.get("draft")
    if not draft:
        print("[critic_draft] no draft in state; accepting empty to terminate")
        return {
            "critic_verdict": "accept",
            "critic_feedback": None,
            "final_briefing": "",
        }

    scored = _rehydrate_scored(state["scored_articles"])
    topic = state["topic"]
    revision_count = state.get("revision_count", 0)

    user_message = (
        f"Topic: {topic}\n\n"
        f"Revision round: {revision_count} "
        f"(0 = first draft, higher means the Writer has already revised)\n\n"
        f"Draft briefing to review:\n\n"
        f"---\n{draft}\n---\n\n"
        f"Source articles (cite only these URLs):\n\n"
        f"{_format_sources_for_prompt(scored)}\n\n"
        f"Review the draft per the rubric and output your verdict."
    )

    agent = build_critic_draft_agent()
    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": user_message}],
    })

    payload = result.get("structured_response")
    if payload is None:
        print("[critic_draft] agent returned no structured_response; falling back to accept")
        return {
            "critic_verdict": "accept",
            "critic_feedback": None,
            "final_briefing": draft,
        }

    # Compact feedback into a single string for the Writer. A list in state
    # would serialize via msgpack fine, but the Writer consumes it as a
    # prompt chunk anyway — pre-formatting here keeps the Writer's
    # prompt-building code simple.
    if payload.verdict == "revise" and payload.feedback_items:
        feedback_text = "\n".join(
            f"- {item}" for item in payload.feedback_items
        )
    else:
        feedback_text = None

    return {
        "critic_verdict": payload.verdict,
        "critic_feedback": feedback_text,
        # Always write final_briefing so force-accept at count=2 produces
        # a valid output without the conditional edge having to touch state.
        "final_briefing": draft,
    }