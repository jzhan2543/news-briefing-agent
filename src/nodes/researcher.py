"""
Researcher node: the only agentic node in the pipeline.

Day 4 stub returns one hardcoded article. Real create_agent implementation
(SPEC §3.2) lands later today.
"""

from src.schemas import Article, BriefingState


def researcher_node_stub(state: BriefingState) -> dict:
    """Stub: returns a single hardcoded Article, preserving source_query provenance."""
    queries = state["search_queries"]
    first_query = queries[0] if queries else "stub"
    return {
        "raw_articles": [
            Article(
                url="https://example.com/stub-article-1",
                title="Stub: Example News Article",
                snippet="This is a stub article used for scaffold testing.",
                published_date=None,
                source_query=first_query,
            ),
        ],
    }


# Real implementation placeholder. Later today:
# - create_agent(model, tools=[web_search], system_prompt, max_iterations=6)
# - RESEARCHER_SYSTEM_PROMPT with {current_date_iso} template
# - def researcher_node(state) -> dict  # loops over search_queries
# - _extract_articles helper for JSON parsing from final AIMessage