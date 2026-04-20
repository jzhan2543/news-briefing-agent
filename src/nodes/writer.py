"""
Writer node: produces the draft briefing from scored articles.

Increments revision_count on every invocation. Count semantics:
  - First Writer call in a run: revision_count goes 0 -> 1 (first draft is "1")
  - Or: treat revision_count as "drafts produced so far". We use the latter
    because it makes the conditional-edge test (revision_count >= 2 forces
    accept) read cleanly: 0 drafts -> run Writer -> 1 draft produced ->
    Critic -> if revise -> Writer again -> 2 drafts produced -> Critic ->
    force-accept regardless of verdict.

Design notes:
- Pretty-prints articles for the prompt, same as Critic(draft). The Writer
  is synthesizing prose; it needs readable input, not dumped JSON.
- Includes CRITIC FEEDBACK block only when critic_feedback is present
  (i.e., on revision rounds). First-draft prompts don't mention revision
  at all — this keeps the first draft's behavior identical to a
  single-agent Formatter, so the multi-agent-vs-single-agent comparison on
  Day 7 can cleanly isolate the feedback loop's contribution.
- Composes final markdown the same way the single-agent Formatter does:
  "# {headline}\n\n{briefing_markdown}". Keeps output format consistent
  across pipelines for side-by-side eval.
"""

from typing import Any

from pydantic import ValidationError

from src.agents.writer import build_writer_agent
from src.schemas import CriticScoredArticle, MultiAgentBriefingState


def _rehydrate_scored(scored_dicts: list[dict]) -> list[CriticScoredArticle]:
    """dicts -> CriticScoredArticle models. Skip malformed entries with a log line.

    Same convention as Filter/Summarizer/Critic(draft). If every entry fails
    validation we still try to produce a draft; the Writer will produce a
    thin briefing or a placeholder, and the Critic will catch the issue.
    Hard-failing here would mean a single malformed state entry kills the
    whole run.
    """
    out: list[CriticScoredArticle] = []
    for d in scored_dicts:
        try:
            out.append(CriticScoredArticle.model_validate(d))
        except ValidationError as e:
            print(f"[writer] skipping malformed scored_article: {e}")
    return out


def _format_sources_for_prompt(scored: list[CriticScoredArticle]) -> str:
    """Render the scored articles into a block the Writer can read and cite from.

    Includes flags because the Writer has to apply flag-aware caveating per
    the system prompt. Format mirrors Critic(draft)'s for consistency in
    prompts the Writer and Critic both see.
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


async def writer_node(state: MultiAgentBriefingState) -> dict[str, Any]:
    """
    Writer node body.

    Reads: state["scored_articles"], state["topic"],
           state["critic_feedback"] (optional, only on revision rounds),
           state["revision_count"] (defaults to 0)
    Writes: draft (the markdown body, headline + briefing),
            revision_count (incremented)

    Degrades gracefully on empty sources: emits a placeholder briefing
    rather than failing. The Critic will review the placeholder and
    likely revise; if sources are empty after revision, force-accept at
    count=2 produces a visible "no sources" final_briefing, which is a
    better UX than raising.
    """
    topic = state["topic"]
    scored = _rehydrate_scored(state["scored_articles"])
    feedback = state.get("critic_feedback")
    revision_count = state.get("revision_count", 0)

    if not scored:
        # Empty sources path. Write a placeholder draft that reads as an
        # honest report rather than a failure. The Critic will see it and
        # either accept (if it's a reasonable placeholder for no-sources)
        # or revise (unlikely to help without sources, but that's the
        # force-accept cap's job to handle).
        placeholder = (
            f"# {topic}\n\n"
            f"_No sufficiently relevant source articles were available for "
            f"this briefing. No claims can be made on this topic given the "
            f"current source set._"
        )
        return {
            "draft": placeholder,
            "revision_count": revision_count + 1,
        }

    sources_block = _format_sources_for_prompt(scored)

    # Build the user message. Revision rounds get a CRITIC FEEDBACK block
    # inserted before the sources; first drafts omit it entirely so first-
    # draft behavior is identical to single-agent Formatter (clean A/B).
    message_parts = [f"Topic: {topic}"]
    if feedback:
        message_parts.append(
            f"\nCRITIC FEEDBACK on the previous draft (revision round "
            f"{revision_count}, address these specifically):\n{feedback}"
        )
    message_parts.append(
        f"\nSource articles (cite only these URLs):\n\n{sources_block}"
    )
    message_parts.append(
        "\nProduce the briefing per the structure and citation requirements."
    )
    user_message = "\n".join(message_parts)

    agent = build_writer_agent()
    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": user_message}],
    })

    payload = result.get("structured_response")
    if payload is None:
        # This really shouldn't happen with response_format on. If it does,
        # treat as a malformed run that the runner's generic except should
        # catch. Raise rather than silently produce a bad draft.
        raise ValueError(
            "[writer] agent returned no structured_response. Check the "
            "LangSmith trace for the final message content."
        )

    final_markdown = f"# {payload.headline}\n\n{payload.briefing_markdown}"
    return {
        "draft": final_markdown,
        "revision_count": revision_count + 1,
    }