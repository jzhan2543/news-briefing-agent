"""Tests for the graph topology in src/graph.py.

These tests assert the shape of the compiled graph: correct node names,
correct edges. They do NOT exercise node logic — that's integration.
"""

from src.graph import build_graph


_EXPECTED_NODES = {
    "__start__",
    "planner",
    "researcher",
    "filter",
    "summarizer",
    "formatter",
    "__end__",
}


_EXPECTED_EDGES = {
    ("__start__", "planner"),
    ("planner", "researcher"),
    ("researcher", "filter"),
    ("filter", "summarizer"),
    ("summarizer", "formatter"),
    ("formatter", "__end__"),
}


def test_build_graph_compiles():
    graph = build_graph()
    assert graph is not None


def test_graph_has_expected_nodes():
    graph = build_graph()
    actual_nodes = set(graph.get_graph().nodes.keys())
    assert actual_nodes == _EXPECTED_NODES


def test_graph_has_expected_edges():
    graph = build_graph()
    actual_edges = {(e.source, e.target) for e in graph.get_graph().edges}
    assert actual_edges == _EXPECTED_EDGES


def test_graph_has_no_conditional_edges_at_outer_layer():
    """Day 4 invariant: outer graph is a linear pipeline with no
    conditional edges. The only model-driven control flow lives inside
    the Researcher's create_agent subgraph. If this test fails, someone
    added branching at the outer layer — which might be intentional,
    but deserves a deliberate spec update."""
    graph = build_graph()
    for edge in graph.get_graph().edges:
        # LangGraph conditional edges have `conditional=True` in their metadata.
        # Plain edges do not.
        assert not getattr(edge, "conditional", False), (
            f"Unexpected conditional edge: {edge.source} -> {edge.target}"
        )