"""
Summarizer node: produces a 2-3 sentence summary per relevant article.

Day 4 stub: produces a trivial summary for every article with relevance >= threshold.
Real implementation (SPEC §3.4) lands on Day 5.
"""

from src.config import RELEVANCE_THRESHOLD
from src.schemas import BriefingState, Summary


def summarizer_node_stub(state: BriefingState) -> dict:
    """Stub: trivial summary for each article above the relevance threshold."""
    return {
        "summaries": [
            Summary(
                article_url=sa.article.url,
                summary=f"Stub summary for: {sa.article.title}",
                key_claims=["Stub claim 1", "Stub claim 2"],
            )
            for sa in state["scored_articles"]
            if sa.relevance >= RELEVANCE_THRESHOLD
        ],
    }


# Real implementation placeholder. Day 5:
# - Pydantic SummaryOutput(summary, key_claims) with length constraints
# - ChatAnthropic(model="claude-sonnet-4-5", temperature=0.0).with_structured_output
# - SUMMARIZER_SYSTEM_PROMPT with "only from snippet" faithfulness rule
# - def summarizer_node(state) -> dict  # one LLM call per eligible article

    