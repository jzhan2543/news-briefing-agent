"""
Planner node: decomposes the topic into 2-4 focused search queries.

Day 4: stub only. Real implementation (from SPEC §3.1) lands on Day 5.
"""

from src.schemas import BriefingState


def planner_node_stub(state: BriefingState) -> dict:
    """Stub: returns two trivial queries derived from the topic."""
    topic = state["topic"]
    return {
        "search_queries": [
            f"{topic} overview",
            f"{topic} latest developments",
        ],
    }


# Real implementation placeholder. Day 5:
# - Pydantic PlannerOutput(queries, rationale)
# - ChatAnthropic(model="claude-sonnet-4-5", temperature=0.3).with_structured_output
# - PLANNER_SYSTEM_PROMPT with {current_date_iso} template
# - def planner_node(state: BriefingState) -> dict