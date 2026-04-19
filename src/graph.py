"""
Graph topology for the news briefing pipeline.

Five nodes, six plain edges, no conditional edges at the outer layer. The
only model-driven control flow lives inside the Researcher's create_agent
subgraph. Everything else is workflow.

This is the Quadrant 2 shape from the SPEC: a deterministic pipeline with
one embedded agentic node. Reading build_graph() tells you what the system
does at a high level in under a minute — which is the whole point of
expressing topology separately from node logic.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.nodes.planner import planner_node_stub
from src.nodes.researcher import researcher_node_stub
from src.nodes.filter import filter_node_stub
from src.nodes.summarizer import summarizer_node_stub
from src.nodes.formatter import formatter_node_stub
from src.schemas import BriefingState


def build_graph(checkpointer=None):
    """Build and compile the briefing pipeline.

    Args:
        checkpointer: Optional LangGraph checkpointer. Defaults to MemorySaver.
            Tests can pass a different checkpointer or None to exercise
            checkpointer-free compilation.

    Returns:
        A compiled LangGraph runnable. Invoke with a dict matching
        BriefingState and a config dict containing a configurable.thread_id.
    """
    graph = StateGraph(BriefingState)

    # Day 4: all five nodes are stubs. Swapped to real implementations
    # one by one starting with the Researcher later today.
    graph.add_node("planner", planner_node_stub)
    graph.add_node("researcher", researcher_node_stub)
    graph.add_node("filter", filter_node_stub)
    graph.add_node("summarizer", summarizer_node_stub)
    graph.add_node("formatter", formatter_node_stub)

    # Linear pipeline: START -> planner -> researcher -> filter ->
    # summarizer -> formatter -> END. No conditional edges.
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "filter")
    graph.add_edge("filter", "summarizer")
    graph.add_edge("summarizer", "formatter")
    graph.add_edge("formatter", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())