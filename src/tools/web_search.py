"""
Web search tool for the Researcher agent.

Wraps Tavily's search API with a tenacity retry policy. The decorator on
the underlying _tavily_search handles transient errors (timeouts, 5xx);
other exceptions propagate so the trace shows the real cause.

The public @tool-decorated function is what the Researcher agent sees. Its
signature, type hints, and docstring are used by create_agent to build the
JSON schema the model calls against — so they're API, not commentary.
"""

import os

import httpx
from langchain_core.tools import tool
from tavily import TavilyClient
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.tools.tavily_cache import cached_tavily_search


# Built once at module load, same pattern as the graph.
_tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(
        (httpx.TimeoutException, httpx.HTTPStatusError)
    ),
    reraise=True,
)
def _tavily_search(query: str, max_results: int = 5) -> list[dict]:
    """Call Tavily and return the results list.

    Retries on transient network failures (timeouts, 5xx). Other errors
    propagate so the LangSmith trace shows the real cause.
    """
    response = _tavily_client.search(query=query, max_results=max_results)
    return response.get("results", [])


@tool
def web_search(query: str) -> str:
    """Search the web for articles relevant to a query.

    Use this to find recent news articles, blog posts, and primary
    sources. Returns up to 5 results as a newline-separated block of
    "title | url | snippet" entries. If no results are found, returns
    an empty string.

    Args:
        query: A short search query (3-8 words works best). Do not
            include year qualifiers unless the user's topic explicitly
            asks about a specific past period.
    """
    # Cached layer: passthrough to _tavily_search when TAVILY_CACHE_DIR
    # is unset (production default); disk-backed record/replay when set
    # (eval harness default).
    results = cached_tavily_search(query, _tavily_search)
    if not results:
        return ""
    lines = []
    for r in results:
        title = r.get("title", "").strip()
        url = r.get("url", "").strip()
        snippet = (r.get("content") or "").strip()
        lines.append(f"{title} | {url} | {snippet}")
    return "\n".join(lines)