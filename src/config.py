"""
Project-wide constants.

Kept in one file so values are auditable and consistently applied across
nodes. If a value is tuned per-run (e.g., different eval runs), promote it
to a function argument rather than editing this file.
"""


# Summarizer boundary: articles scoring below this are dropped and not
# summarized. The rubric defines 3 as "mentions the topic but primarily
# about something else" — below that is not worth the summary cost.
RELEVANCE_THRESHOLD: int = 3


# Hard cap on Tavily search calls per run_briefing() invocation. Protects
# against cost blowouts from a misbehaving Researcher. The Researcher's
# create_agent(max_iterations=6) is a separate, lower-level safety cap on
# agent-loop iterations; this constant is the budget-level cap across all
# Researcher invocations in a single run. Enforced starting Day 5 (when
# async fan-out makes exhausting the quota plausible).
MAX_SEARCH_CALLS_PER_RUN: int = 6