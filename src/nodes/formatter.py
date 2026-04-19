"""
Formatter node: assembles summaries into a final markdown briefing.

Day 4 stub: produces minimal markdown with a header and bullet list.
Real implementation (SPEC §3.5) lands on Day 5.
"""

from src.schemas import BriefingState, Summary


def formatter_node_stub(state: BriefingState) -> dict:
    """Stub: bare-minimum markdown skeleton."""
    topic = state["topic"]
    lines = [
        f"# {topic} — Stub Briefing",
        "",
        "Stub briefing assembled from the following sources:",
        "",
    ]
    for s_dict in state["summaries"]:
        s = Summary.model_validate(s_dict)
        lines.append(f"- {s.article_url}: {s.summary}")
    return {"final_briefing": "\n".join(lines)}


# Real implementation placeholder. Day 5:
# - ChatAnthropic(model="claude-sonnet-4-5", temperature=0.2)  # no structured output
# - FORMATTER_SYSTEM_PROMPT with citation-required rule and structure template
# - def formatter_node(state) -> dict  # single LLM call on all summaries