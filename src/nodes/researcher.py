"""
Researcher node: the only agentic node in the pipeline.

Wraps a create_agent sub-agent as a node in the outer graph. This is the
canonical Quadrant 2 pattern — agent inside a workflow. The sub-agent uses
MessagesState internally; we translate its final output to list[Article]
dicts for the outer BriefingState.
"""

from datetime import datetime

import asyncio

from langchain.agents import create_agent

from src.schemas import Article, BriefingState
from src.tools.web_search import web_search
from src.llm import get_chat_model, RESEARCHER_MODEL


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
"""

from pydantic import BaseModel, Field

class _ArticlePayload(BaseModel):
    """One article as emitted by the research agent. No source_query here;
    we stitch that in on the way out since the agent doesn't know which
    planner query invoked it."""
    url: str
    title: str
    snippet: str
    published_date: str | None = None


class _ResearcherOutput(BaseModel):
    articles: list[_ArticlePayload] = Field(
        min_length=1,
        max_length=5,
        description="3-5 articles found via web_search, most relevant first.",
    )

def _build_researcher_agent(current_date_iso: str):
    """Build the researcher sub-agent. Called once per run_briefing invocation."""
    return create_agent(
        # model=ChatAnthropic(model="claude-sonnet-4-5", temperature=0.0),
        model=get_chat_model(model=RESEARCHER_MODEL),
        tools=[web_search],
        system_prompt=_RESEARCHER_SYSTEM_PROMPT_TEMPLATE.format(
            current_date_iso=current_date_iso,
        ),
        response_format=_ResearcherOutput,
    )


def _extract_articles(agent_result: dict, source_query: str) -> list[Article]:
    """Read the agent's structured response and stitch in source_query.

    With response_format, the agent's final step is a schema-coerced call,
    so no text parsing is needed and no string-escape edge cases apply.
    """
    payload: _ResearcherOutput = agent_result["structured_response"]
    return [
        Article(
            url=p.url,
            title=p.title,
            snippet=p.snippet,
            published_date=p.published_date,
            source_query=source_query,
        )
        for p in payload.articles
    ]

async def _research_one_query(agent, topic: str, query: str) -> list[Article]:
    """Run the researcher sub-agent for a single planner query.

    Pulled out of researcher_node so asyncio.gather can fan out over it.
    Raises on agent-level failures (schema violations, missing tools);
    caller decides whether to tolerate per-query failures or propagate.
    """
    result = await agent.ainvoke({
        "messages": [{
            "role": "user",
            "content": (
                f"Topic under investigation: {topic}\n"
                f"Run web searches to find articles relevant to: {query}"
            ),
        }],
    })
    return _extract_articles(result, source_query=query)


async def researcher_node(state: BriefingState) -> dict:
    """Invoke the researcher sub-agent once per search query, in parallel.

    Returns articles as dicts for msgpack-compatible state storage.
    Validation happens at Article construction time inside _extract_articles.

    Per-query failures are logged and skipped rather than killing the
    whole run — same convention as Filter and Summarizer. If every query
    fails, raw_articles comes back empty; downstream nodes handle empty
    input gracefully (Filter returns []; Formatter falls back to a
    "no articles" placeholder).
    """
    current_date_iso = state["run_started_at"].date().isoformat()
    agent = _build_researcher_agent(current_date_iso)

    # Real parallelism: one agent invocation per planner query, all running
    # concurrently. In LangSmith the per-query spans overlap on the timeline
    # rather than stacking end-to-end.
    results = await asyncio.gather(
        *(
            _research_one_query(agent, state["topic"], query)
            for query in state["search_queries"]
        ),
        return_exceptions=True,
    )

    articles: list[Article] = []
    for query, result in zip(state["search_queries"], results):
        if isinstance(result, Exception):
            print(f"[researcher] query failed for {query!r}: {result}")
            continue
        articles.extend(result)

    return {"raw_articles": [a.model_dump(mode="json") for a in articles]}