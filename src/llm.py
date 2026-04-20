"""Shared LLM client factory and model configuration for the briefing pipeline.

One place for model names so eval sweeps on Day 6 can change them without
touching every node. We use ChatAnthropic throughout (not the raw Anthropic
SDK) for consistent LangSmith tracing and compatibility with LangChain's
create_agent.
"""

from langchain_anthropic import ChatAnthropic


DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_TEMPERATURE = 0.0

# Per-node model overrides. All default to DEFAULT_MODEL today; Day 6 eval
# work may sweep these independently (e.g. cheaper model for Filter scoring).
RESEARCHER_MODEL = DEFAULT_MODEL
FILTER_MODEL = DEFAULT_MODEL
SUMMARIZER_MODEL = DEFAULT_MODEL
FORMATTER_MODEL = DEFAULT_MODEL
PLANNER_MODEL = DEFAULT_MODEL


def get_chat_model(model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE) -> ChatAnthropic:
    """Construct a ChatAnthropic instance. Factory rather than module-level
    singleton so each node can override params (temperature, max_tokens)
    without mutating shared state."""
    return ChatAnthropic(model=model, temperature=temperature)