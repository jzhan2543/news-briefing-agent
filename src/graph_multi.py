"""
Graph topology for the multi-agent news briefing pipeline.

Five nodes, six edges, one conditional edge. The supervisor pattern — but
the "supervisor" itself is a conditional edge, not an LLM node. The only
genuinely model-driven control-flow decision is the Critic(draft)'s
accept/revise verdict; everything else is deterministic routing.

This is the Quadrant 4 shape from the SPEC: a workflow with multiple
embedded agents coordinating via a state machine. Contrast with
`src/graph.py` (Quadrant 2: linear workflow with one embedded agent).

Topology:
  START
    |
  planner
    |
  researcher           (parallel over queries, reused as-is)
    |
  critic_articles      (replaces filter; adds structured flags)
    |
  writer               (increments revision_count on every call)
    |
  critic_draft         (always writes final_briefing = draft)
    |
  [conditional edge: should_continue_revising]
    |-- accept OR revision_count >= 2  --> END
    |-- revise AND revision_count < 2  --> writer (loop)

The single-agent graph uses summaries + formatter; the multi-agent graph
skips summaries (the Writer synthesizes final markdown directly). The
inherited summaries field on MultiAgentBriefingState goes unused —
acceptable dead weight for now; documented in SPEC and flagged as a
Day 6 cleanup candidate.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.nodes.planner import planner_node
from src.nodes.researcher import researcher_node
from src.nodes.critic_articles import critic_articles_node
from src.nodes.critic_draft import critic_draft_node
from src.nodes.writer import writer_node
from src.schemas import MultiAgentBriefingState


# Revision cap. Tuned to 2 so the Writer gets one first-draft chance plus
# one revision chance; after that, force-accept regardless of verdict to
# bound cost. If the Critic rejects every draft, we exit with whatever the
# Writer's second draft was. This is a behavioral guardrail — documented
# in SPEC and surfaced as a metric (force-accept rate) in Day 6 evals.
_REVISION_CAP = 2


def should_continue_revising(state: MultiAgentBriefingState) -> str:
    """Conditional edge function — the one model-driven routing decision.

    Order matters:
    1. Accept short-circuits regardless of count. A first-draft accept
       (count=1) must exit, not loop.
    2. Count cap catches the force-accept case. Exits even on a "revise"
       verdict once the budget is spent.
    3. Otherwise (revise + under cap), loop back to Writer for another
       revision round.

    Anything that isn't explicitly "revise" is treated as an exit condition.
    This means if Critic(draft) falls back to a safe default (e.g., on
    malformed agent output), the graph terminates cleanly rather than
    looping forever.
    """
    if state.get("critic_verdict") == "accept":
        return END
    if state.get("revision_count", 0) >= _REVISION_CAP:
        return END
    if state.get("critic_verdict") == "revise":
        return "writer"
    # Defensive fallback: any other state (None verdict, unexpected value)
    # exits cleanly. final_briefing is always populated by Critic(draft)
    # regardless of verdict, so this is safe.
    return END


def build_multi_agent_graph(checkpointer=None):
    """Build and compile the multi-agent briefing pipeline.

    Args:
        checkpointer: Optional LangGraph checkpointer. Defaults to MemorySaver.
            Matches the single-agent graph's signature for symmetry.

    Returns:
        A compiled LangGraph runnable. Invoke with a dict matching
        MultiAgentBriefingState and a config dict containing a
        configurable.thread_id.
    """
    graph = StateGraph(MultiAgentBriefingState)

    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("critic_articles", critic_articles_node)
    graph.add_node("writer", writer_node)
    graph.add_node("critic_draft", critic_draft_node)

    # Linear spine: START -> planner -> researcher -> critic_articles -> writer
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "critic_articles")
    graph.add_edge("critic_articles", "writer")

    # Writer -> Critic(draft) is a plain edge: we always review every draft.
    graph.add_edge("writer", "critic_draft")

    # Critic(draft) -> conditional. The one model-driven branch in the graph.
    # Branch targets must be node names as strings; END is a sentinel.
    graph.add_conditional_edges(
        "critic_draft",
        should_continue_revising,
        {
            "writer": "writer",
            END: END,
        },
    )

    return graph.compile(checkpointer=checkpointer or MemorySaver())