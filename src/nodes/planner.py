"""
Planner node: decomposes the input topic into 3 search queries, each
targeting a distinct sub-aspect of the topic.

Design notes:
- Exactly 3 queries (not a range): predictable downstream cost and
  easier eval reproducibility. If the topic genuinely can't yield 3
  distinct queries, that's a signal the topic is too narrow to brief
  well; Pydantic's min_length=3 surfaces that as a schema_violation
  via the runner rather than silently producing a thin briefing.
- Distinct sub-aspects, not synonym variations. Synonym-style query
  expansion (e.g., "EU AI regulation" / "European AI law") returns
  nearly identical article sets and wastes tokens; sub-aspect expansion
  (e.g., enforcement / compliance / general-purpose models) produces
  thematically richer raw_articles for the downstream Filter and
  Summarizer to work with.
- Rationales are generated alongside queries but dropped before the
  state write. Keeping them in the LLM call acts as a forcing function:
  three rationales are harder to mush together than three queries, so
  the model actually commits to distinct angles. They live in the
  LangSmith trace for debugging but do not clutter BriefingState since
  nothing downstream consumes them.
- Current date injected into the prompt so queries don't hard-code
  stale years — same fix as the Researcher's Day 2 staleness bug.
- Temperature 0.0: planning is structured decomposition, not creative
  writing; determinism matters for eval reproducibility.
"""

from typing import Any

from pydantic import BaseModel, Field

from src.llm import PLANNER_MODEL, get_chat_model
from src.schemas import BriefingState


class _PlannedQuery(BaseModel):
    """One search query with the rationale that justifies it.

    The rationale is a forcing function for distinct angles at generation
    time; it is dropped before the state write (see module docstring).
    """

    query: str = Field(
        min_length=3,
        description="A web search query, 3-10 words. No quotes, no site: operators.",
    )
    rationale: str = Field(
        min_length=10,
        description=(
            "One sentence explaining which sub-aspect of the topic this "
            "query targets. Must describe a different dimension than the "
            "other queries' rationales."
        ),
    )


class _PlannerOutput(BaseModel):
    """Structured output from the Planner LLM call."""

    queries: list[_PlannedQuery] = Field(
        min_length=3,
        max_length=3,
        description=(
            "Exactly 3 planned queries, each targeting a distinct "
            "sub-aspect of the topic. No synonym variations."
        ),
    )


_model = get_chat_model(model=PLANNER_MODEL, temperature=0.0)
_structured_model = _model.with_structured_output(_PlannerOutput)


_PLANNER_SYSTEM_PROMPT_TEMPLATE = """\
You are a research planner. Given a topic, your job is to produce exactly
3 web search queries that together give comprehensive coverage of the topic.

Today's date is {current_date_iso}. Use this when judging what "recent"
means. Do not hard-code specific years in queries unless the topic
explicitly asks about a past time period.

Rules for the 3 queries:
- Each query must target a DISTINCT sub-aspect of the topic. Think about
  what dimensions a thorough briefing would need to cover — for example,
  current events, underlying policy or mechanics, enforcement or impact,
  stakeholder reactions, historical context. Pick 3 that fit the topic.
- Do NOT produce synonym variations. "EU AI regulation" and "European AI
  law" target the same documents and waste tokens. If two of your queries
  would return overlapping article sets, rewrite one of them.
- Keep queries short and direct (3-10 words). No quoted phrases. No
  site: or date: operators. No years unless the topic requires it.
- Each query must be paired with a one-sentence rationale describing
  which sub-aspect it targets. The rationales must describe different
  dimensions.
"""


async def planner_node(state: BriefingState) -> dict[str, Any]:
    """
    Planner node body.

    Reads: state["topic"], state["run_started_at"]
    Writes: search_queries (list[str] — rationales dropped)

    Raises: ValidationError (caught by runner) if the model fails to
    produce exactly 3 distinct-looking queries after structured-output
    retries. Surfaces as BriefingFailure(reason="schema_violation").
    """
    topic = state["topic"]
    current_date_iso = state["run_started_at"].date().isoformat()

    system_prompt = _PLANNER_SYSTEM_PROMPT_TEMPLATE.format(
        current_date_iso=current_date_iso,
    )
    user_message = (
        f"Topic: {topic}\n\n"
        f"Produce 3 distinct search queries covering different sub-aspects "
        f"of this topic, each with a one-sentence rationale."
    )

    payload: _PlannerOutput = await _structured_model.ainvoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
    )

    # Drop rationales; only the query strings flow downstream. The full
    # structured output is preserved in the LangSmith trace for debugging.
    return {"search_queries": [pq.query for pq in payload.queries]}