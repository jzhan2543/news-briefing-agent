"""
Day 3, Pass 2: ReAct agent as an explicit StateGraph (LangGraph layer).

Same behavior as agent_v1_prebuilt.py, but with the graph built by hand.
No create_agent — we write the model node, the tools node, and the
conditional edge ourselves. This is what create_agent was hiding.
"""
from dotenv import load_dotenv
from typing import Literal

from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from tavily import TavilyClient
from datetime import datetime, timezone
import os

load_dotenv()


# --- Tools (identical to Pass 1) ---

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


tools = [web_search, get_current_time]


# --- Model, bound to the tools ---

SYSTEM_PROMPT = (
    "You are a helpful research assistant. Use the web_search tool when you "
    "need current information, and get_current_time when the user asks about "
    "time or dates. If search results don't clearly answer the question, say "
    "so rather than guessing."
)

model = ChatAnthropic(model="claude-sonnet-4-5").bind_tools(tools)


# --- Nodes ---
def call_model(state: MessagesState):
    """The model node. Takes current messages, calls the LLM, returns the
    AI response as a partial state update."""
    messages = state["messages"]
    # Prepend the system prompt if this is the first call
    if not any(m.type == "system" for m in messages):
        from langchain_core.messages import SystemMessage
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = model.invoke(messages)
    return {"messages": [response]}


# ToolNode is a pre-built node that executes any pending tool calls
# in the last AI message and returns ToolMessage results.
tool_node = ToolNode(tools)


# --- Conditional edge function ---
def should_continue(state: MessagesState):
    """Route based on whether the last AI message contains tool calls.
    If yes -> tools node. If no -> END."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# --- Build the graph --- 
## NOTES: literally building the ReAct but in Langgraph (2 nodes 3 edges)

graph_builder = StateGraph(MessagesState)

graph_builder.add_node("model", call_model)          # node named "model" runs call_model
graph_builder.add_node("tools", tool_node)           # node named "tools" runs tool_node
graph_builder.add_edge(START, "model")               # start → model
graph_builder.add_conditional_edges("model", should_continue)  # model → (tools or END)
graph_builder.add_edge("tools", "model")             # tools → model (plain, always loops back)

graph = graph_builder.compile()


# --- Run ---

if __name__ == "__main__":
    query = "What's the latest news on AI regulation?"
    result = graph.invoke({"messages": [{"role": "user", "content": query}]})
    
    final_message = result["messages"][-1]
    print(f"\n=== FINAL ANSWER ===\n{final_message.content}\n")
    
    print("=== TRAJECTORY ===")
    for msg in result["messages"]:
        print(f"[{msg.type}] {str(msg.content)[:200]}...")