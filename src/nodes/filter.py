"""
Filter node: scores each article's relevance to the topic.

Day 4 stub: gives every article a relevance of 4 without thinking.
Real implementation (SPEC §3.3) lands on Day 5.
"""

from src.schemas import Article, BriefingState, ScoredArticle


def filter_node_stub(state: BriefingState) -> dict:
    """Stub: uniform relevance=4 for every article."""
    scored = []
    for article_dict in state["raw_articles"]:
        article = Article.model_validate(article_dict)
        sa = ScoredArticle(article=article, relevance=4, rationale="stub")
        scored.append(sa.model_dump(mode="json"))
    return {"scored_articles": scored}

# Real implementation placeholder. Day 5:
# - Pydantic FilterJudgment(relevance, rationale)
# - ChatAnthropic(model="claude-haiku-4-5", temperature=0.0).with_structured_output
# - FILTER_SYSTEM_PROMPT with 1-5 rubric
# - def filter_node(state) -> dict  # one LLM call per raw article