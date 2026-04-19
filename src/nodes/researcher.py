"""
Researcher node: the only agentic node in the pipeline.

Wraps a create_agent sub-agent as a node in the outer graph. This is the
canonical Quadrant 2 pattern — agent inside a workflow. The sub-agent uses
MessagesState internally; we translate its final output to list[Article]
dicts for the outer BriefingState.
"""

import json
from datetime import datetime

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic

from src.schemas import Article, BriefingState
from src.tools.web_search import web_search


_RESEARCHER_SYSTEM_PROMPT_TEMPLATE = """\
You are a research agent. Your job is to find recent, credible web articles
on a given topic using the web_search tool.

Today's date is {current_date_iso}. Use this date when judging recency.
Do not include specific years in search queries unless the user's query
explicitly asks about a past time period.

Guidelines:
- Start with the user's query as given. Run the search.
- If the first search returns fewer than 3 relevant results, refine the
  query and search again. Do not exceed 3 search calls per invocation.
- Prefer established news and primary sources over aggregators.
- Do not invent URLs, dates, or snippets. Every article you return must
  come from an actual search result.

When you have 3-5 relevant results, respond with ONLY a JSON array. No
preamble, no trailing commentary, no markdown fencing. Schema per item:

  {{
    "url": str,
    "title": str,
    "snippet": str,
    "published_date": str | null
  }}
"""


def _build_researcher_agent(current_date_iso: str):
    """Build the researcher sub-agent. Called once per run_briefing invocation."""
    return create_agent(
        model=ChatAnthropic(model="claude-sonnet-4-5", temperature=0.0),
        tools=[web_search],
        system_prompt=_RESEARCHER_SYSTEM_PROMPT_TEMPLATE.format(
            current_date_iso=current_date_iso,
        ),
    )


def _extract_articles(agent_result: dict, source_query: str) -> list[Article]:
    """Parse the final AIMessage's content as a JSON array of articles.

    Fails loud: any parse error or schema error raises. The trace will
    show the exact agent output that couldn't be parsed.
    """
    final_message = agent_result["messages"][-1]
    payload = json.loads(final_message.content)
    return [
        Article(**{**item, "source_query": source_query})
        for item in payload
    ]


def researcher_node(state: BriefingState) -> dict:
    """Invoke the researcher sub-agent once per search query.

    Returns articles as dicts for msgpack-compatible state storage.
    Validation happens at Article construction time inside _extract_articles.
    """
    current_date_iso = state["run_started_at"].date().isoformat()
    agent = _build_researcher_agent(current_date_iso)

    articles: list[Article] = []
    for query in state["search_queries"]:
        result = agent.invoke({
            "messages": [{
                "role": "user",
                "content": (
                    f"Topic under investigation: {state['topic']}\n"
                    f"Run web searches to find articles relevant to: {query}\n"
                    f"When you have 3-5 relevant results, return them as a "
                    f"JSON array of objects with keys: url, title, snippet, "
                    f"published_date (ISO 8601 or null). Do not include any "
                    f"commentary outside the JSON."
                ),
            }],
        })
        articles.extend(_extract_articles(result, source_query=query))

    return {"raw_articles": [a.model_dump(mode="json") for a in articles]}