"""Tests for the multi-agent graph topology in src/graph_multi.py.

Same shape discipline as test_graph_topology.py: these tests assert the
compiled graph's node and edge structure, not node logic.

The multi-agent graph differs from the single-agent graph in three
structural ways that should be visible in the test:
  1. Different node set: no filter/summarizer/formatter; add critic_articles,
     writer, critic_draft.
  2. One conditional edge from critic_draft (the only model-driven branch).
  3. The Writer -> critic_draft -> (writer | END) cycle — writer is a
     multi-entry node (reached from both critic_articles and the conditional
     edge loop).

The single-agent test asserts NO conditional edges at the outer layer;
this test's mirror asserts EXACTLY ONE conditional edge, anchored to
critic_draft. If someone adds a second conditional edge, the assert
fires and we audit the change.
"""

from langgraph.graph import END

from src.graph_multi import build_multi_agent_graph, should_continue_revising


_EXPECTED_NODES = {
    "__start__",
    "planner",
    "researcher",
    "critic_articles",
    "writer",
    "critic_draft",
    "__end__",
}


# Plain (non-conditional) edges only. The critic_draft -> writer loop-back
# and critic_draft -> END exits live on the conditional edge, not here.
_EXPECTED_PLAIN_EDGES = {
    ("__start__", "planner"),
    ("planner", "researcher"),
    ("researcher", "critic_articles"),
    ("critic_articles", "writer"),
    ("writer", "critic_draft"),
}


def test_multi_agent_graph_compiles():
    graph = build_multi_agent_graph()
    assert graph is not None


def test_multi_agent_graph_has_expected_nodes():
    graph = build_multi_agent_graph()
    actual_nodes = set(graph.get_graph().nodes.keys())
    assert actual_nodes == _EXPECTED_NODES


def test_multi_agent_graph_plain_edges_match():
    """Assert the deterministic spine of the graph. Conditional edges are
    checked separately below."""
    graph = build_multi_agent_graph()
    plain_edges = {
        (e.source, e.target)
        for e in graph.get_graph().edges
        if not getattr(e, "conditional", False)
    }
    assert plain_edges == _EXPECTED_PLAIN_EDGES


def test_multi_agent_graph_has_exactly_one_conditional_edge_source():
    """Multi-agent invariant: the graph has exactly one model-driven branch
    point, and that point is critic_draft. If this test fails, someone added
    a second model-driven decision to the outer graph — which might be
    intentional but deserves a deliberate spec update.
    """
    graph = build_multi_agent_graph()
    conditional_sources = {
        e.source
        for e in graph.get_graph().edges
        if getattr(e, "conditional", False)
    }
    assert conditional_sources == {"critic_draft"}, (
        f"Expected exactly one conditional edge from critic_draft; "
        f"got {conditional_sources}"
    )


def test_conditional_edge_accept_routes_to_end():
    """should_continue_revising returns END on 'accept' regardless of count.
    This is the 'first-draft accept exits immediately' invariant — without
    it, a clean first draft would loop unnecessarily."""
    state = {
        "critic_verdict": "accept",
        "revision_count": 1,
    }
    assert should_continue_revising(state) == END


def test_conditional_edge_revise_under_cap_routes_to_writer():
    """should_continue_revising returns 'writer' on revise + count under cap."""
    state = {
        "critic_verdict": "revise",
        "revision_count": 1,
    }
    assert should_continue_revising(state) == "writer"


def test_conditional_edge_revise_at_cap_routes_to_end():
    """should_continue_revising force-accepts at revision_count >= 2 even
    on a 'revise' verdict. This is the cost-bounding guardrail."""
    state = {
        "critic_verdict": "revise",
        "revision_count": 2,
    }
    assert should_continue_revising(state) == END


def test_conditional_edge_unexpected_verdict_routes_to_end():
    """Defensive fallback: any verdict that isn't explicitly 'revise' exits
    cleanly. Covers None (shouldn't happen, but could on malformed Critic
    output) and any hypothetical future values."""
    state = {
        "critic_verdict": None,
        "revision_count": 0,
    }
    assert should_continue_revising(state) == END