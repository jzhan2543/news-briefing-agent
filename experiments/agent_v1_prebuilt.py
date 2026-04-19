"""
Day 3, Pass 1: ReAct agent via LangChain's create_agent (LangChain layer).

Same behavior as agent_v0.py, built on the LangChain prebuilt abstraction.
No explicit graph construction — create_agent returns a compiled LangGraph
runnable that implements the standard ReAct loop.
"""
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from tavily import TavilyClient
from datetime import datetime, timezone
import os

load_dotenv()

# @tool replaces the need for json schema?? thats crazy lol 
@tool
def web_search(query: str):
    """Search the web for current information. Use this when the user asks
    about recent events, news, or anything that might have changed recently."""
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = client.search(query=query, max_results=5)
    return "\n\n".join(
        f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content']}"
        for r in results["results"]
    )


@tool
def get_current_time():
    """Returns the current UTC time in ISO 8601 format. Use this when the user
    asks about the current time, today's date, or anything where 'now' matters."""
    return datetime.now(timezone.utc).isoformat()

# --- Agent ---
SYSTEM_PROMPT = (
    "You are a helpful research assistant. Use the web_search tool when you "
    "need current information, and get_current_time when the user asks about "
    "time or dates. If search results don't clearly answer the question, say "
    "so rather than guessing."
)

# bruh could just be using to create agent this whole time? 
# notes: the loop is baked into the create agent, we just hand it a model, toosl, and prompt
agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[web_search, get_current_time],
    system_prompt=SYSTEM_PROMPT,
)

if __name__ == "__main__":
    query = "What's the latest news on AI regulation?"
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    
    # The final answer is the last message in the returned state
    final_message = result["messages"][-1]
    print(f"\n=== FINAL ANSWER ===\n{final_message.content}\n")
    
    # Optional: inspect the full trajectory
    print("=== TRAJECTORY ===")

    # so the messages here are akin to the messages=[{"role": "user", "content": "..."}]
    for msg in result["messages"]:
        print(f"[{msg.type}] {str(msg.content)[:200]}...")