"""
Top-level wrapper for the briefing pipeline.

run_briefing(topic) is the boundary between two different error-handling
regimes: graph nodes fail loud (raise on any validation or parse error),
but callers of run_briefing() never see a stack trace. Exceptions thrown
inside the graph are caught here, classified into a BriefingFailure.reason,
and returned as data alongside the thread_id for LangSmith trace lookup.

See SPEC §5 for the full contract and the rationale for the interim
`except Exception` / reason="unknown" posture.
"""

from datetime import UTC, datetime
from uuid import uuid4

from langgraph.errors import GraphRecursionError
from pydantic import ValidationError

from src.graph import build_graph
from src.schemas import BriefingFailure, BriefingResult, BriefingSuccess


# Input validation thresholds. Topics shorter than 15 chars are too vague
# to plan against; topics longer than 500 chars are not topics, they're
# paragraphs. Documented in SPEC §5 under Input.
_MIN_TOPIC_LEN = 15
_MAX_TOPIC_LEN = 500

# Built once at module load. Matches the SPEC §5 choice — production-style
# pattern, testable because build_graph() accepts a checkpointer arg for
# tests that want to swap it out.
_compiled_graph = build_graph()


def run_briefing(topic: str) -> BriefingResult:
    """Run the briefing pipeline for one topic.

    Never raises; always returns a BriefingResult. Failures inside the
    graph are caught and classified. The thread_id is preserved on both
    success and failure so the LangSmith trace is always recoverable.
    """
    started_at = datetime.now(UTC)

    # --- input validation (before graph invocation) ---
    if not isinstance(topic, str):
        return BriefingFailure(
            topic=str(topic),
            reason="invalid_topic",
            message=f"Expected str, got {type(topic).__name__}.",
            run_started_at=started_at,
            run_failed_at=datetime.now(UTC),
            thread_id=None,
        )
    topic = topic.strip()
    if not (_MIN_TOPIC_LEN <= len(topic) <= _MAX_TOPIC_LEN):
        return BriefingFailure(
            topic=topic,
            reason="invalid_topic",
            message=(
                f"Topic must be {_MIN_TOPIC_LEN}-{_MAX_TOPIC_LEN} chars; "
                f"got {len(topic)}."
            ),
            run_started_at=started_at,
            run_failed_at=datetime.now(UTC),
            thread_id=None,
        )

    # --- graph invocation ---
    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "topic": topic,
        "run_started_at": started_at,
        "search_queries": [],
        "raw_articles": [],
        "scored_articles": [],
        "summaries": [],
        "final_briefing": "",
    }

    try:
        final_state = _compiled_graph.invoke(initial_state, config=config)
    except GraphRecursionError as e:
        return BriefingFailure(
            topic=topic,
            reason="max_iterations",
            message=f"Agent exceeded max iterations: {e}",
            run_started_at=started_at,
            run_failed_at=datetime.now(UTC),
            thread_id=thread_id,
        )
    except ValidationError as e:
        return BriefingFailure(
            topic=topic,
            reason="schema_violation",
            message=f"A node produced output that failed validation: {e}",
            run_started_at=started_at,
            run_failed_at=datetime.now(UTC),
            thread_id=thread_id,
        )
    except Exception as e:
        # Last-resort catch. Every other failure mode above is structured.
        # Any "unknown" result here is effectively a bug report; the plan
        # is to promote recurring ones to named reasons after Day 6 evals
        # show what actually fires in practice.
        return BriefingFailure(
            topic=topic,
            reason="unknown",
            message=f"{type(e).__name__}: {e}",
            run_started_at=started_at,
            run_failed_at=datetime.now(UTC),
            thread_id=thread_id,
        )

    # --- success path ---
    return BriefingSuccess(
        topic=topic,
        briefing_markdown=final_state["final_briefing"],
        run_started_at=started_at,
        run_completed_at=datetime.now(UTC),
        thread_id=thread_id,
        source_urls=[s["article_url"] for s in final_state["summaries"]],
    )