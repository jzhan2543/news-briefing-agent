# News Briefing Agent — Specification

**Status:** Locked, end of Day 4 planning. Draft reflects decisions made in Day 4 spec-drafting session.

---

## 1. Purpose & Scope

### Purpose

Given a user-supplied topic (e.g., "AI regulation", "Fed rate decisions"), produce a short, cited markdown briefing summarizing recent web coverage. The system exists as a learning vehicle for agentic architecture — specifically, to exercise a Quadrant 2 shape (explicit `StateGraph` with custom state, containing one `create_agent`-based agentic node) and to generate a concrete artifact for interview discussion.

### In scope

- Single user, single run, synchronous on Day 4 scaffold; async upgrade path from Day 5 onward (parallel Researcher queries, fan-out Filter/Summarizer).
- English-language web content only.
- Snippet-based summarization (upgrade path to full-page fetch documented below).
- Observability via LangSmith tracing — every node and every tool call visible as spans.
- Day 5 adds a multi-agent (supervisor) variant producing the same shape of output for comparison.
- Day 6 adds a golden-dataset eval harness with three metrics (source quality, trajectory, faithfulness).
- Day 7 adds guardrails (input validation, max-iteration cap, citation-required check) and one HITL interrupt.

### Out of scope (this week)

- Authentication, user accounts, persistence beyond one run.
- Daily scheduling, email delivery, topic selection automation, cross-run deduplication. See "Extensions Not Built This Week" below.
- Full-page fetching and HTML cleaning (upgrade path, not required to ship).
- PDF / paywall / login-required sources.
- Non-English content.
- Multi-turn conversation. The system is one-shot: topic in, briefing out. A `MemorySaver` checkpointer is wired in from Day 3, but is used for state inspection rather than conversational continuation.

### Success criteria

A run is *successful* if:

1. The pipeline terminates cleanly (no exception, no max-iteration timeout).
2. The output is valid markdown with at least one cited source.
3. Every claim in the final briefing has a citation URL that actually appeared in the retrieved articles.

Note: #3 is aspirational for the Day 4 stub scaffold (stubs don't produce real claims). It becomes the canonical correctness property once the Formatter is real on Day 5, and is enforced by a Day 7 output guardrail.

A run is *high-quality* (measured on Day 6) if:

1. Source quality ≥ 0.8 (cited domains are in the reputable list for the topic).
2. Trajectory is sensible (Researcher ran 2–4 searches, didn't loop, didn't short-circuit).
3. Faithfulness ≥ 4/5 on the LLM-as-judge rubric across the briefing's claims.

### Extensions Not Built This Week

The following extensions were considered and deliberately deferred to keep the week focused on agentic architecture fundamentals. Each is sketched in enough detail to show the design was understood, not ignored.

#### Daily personal briefing pipeline

**Goal:** a scheduled job that runs each morning, produces a personalized briefing on US news and technology (with urgent items flagged), and delivers it by email.

**Architecture sketch:**

```
[Scheduler]                          ← cron / GitHub Actions / cloud scheduler
    ↓
[Topic Planner]                      ← LLM call: propose 3-5 topics for today
    ↓ (one invocation per topic, parallelized)
[Existing News Briefing Pipeline]    ← this project, unchanged
    ↓
[Deduplication layer]                ← compare URLs/claims vs. last N days
    ↓
[Urgency Classifier]                 ← LLM call: flag breaking / time-sensitive
    ↓
[Email Formatter + Delivery]         ← SendGrid / Postmark / SES
    ↓
[Run Store]                          ← persist topics, URLs, outputs for dedup
```

**What's hard about this that the core project doesn't exercise:**

- *Topic selection.* The current pipeline takes a topic as input. A daily digest has to decide what today's topics *are*, which is itself an agentic problem with its own failure modes.
- *Deduplication across runs.* Requires a persistent store and a notion of "we already covered this story yesterday." Either URL-level (cheap, misses restatements) or claim-level (expensive, needs embeddings or LLM comparison).
- *Urgency classification.* A well-known hard problem. LLM-as-classifier works but has its own calibration concerns; would need its own eval.
- *Silent failures.* Scheduled jobs that fail at 7am are discovered at noon. Production-grade would need alerting, dead-letter queues, and retry policies — none of which is the learning goal this week.
- *Email deliverability.* SPF, DKIM, DMARC, bounce handling — ops, not AI.

**Why scoped out:** none of the above is the agentic AI skill this sprint is building. The core pipeline, eval harness, and multi-agent refactor are. Revisiting after Day 7.

#### Full-page article fetch (upgrade from snippets)

*Architecture:* post-Researcher node that fetches, cleans (readability/boilerplate strip), and truncates each article's URL. Replaces the `snippet` field in the `Article` schema with full text.

*Why scoped out:* adds an I/O-bound fan-out step, introduces failure modes (timeouts, 403s, paywalls), and the Day 6 eval will tell us empirically whether snippets are insufficient — a data-driven upgrade decision is stronger than an upfront one.

#### Vector-store-backed long-term memory

*Architecture:* a Chroma/pgvector index of past briefings and their citations, queried as a retrieval tool during the Researcher step to surface "what have we already said about this topic."

*Why scoped out:* long-term memory is the right answer for a production system, but the Day 2–3 build has already established short-term memory via `MessagesState`. Adding a vector store would blow the week budget without adding a proportionate learning increment.

---

## 2. State Schema

The outer graph uses a custom `TypedDict` — not the pre-built `MessagesState` — because the pipeline carries structured artifacts between nodes (articles, scores, summaries) rather than a conversation history. Each state key has an explicit reducer chosen for how that key accumulates across node updates.

### `BriefingState`

```python
from typing import TypedDict, Annotated
from operator import add
from pydantic import BaseModel, Field, HttpUrl
from datetime import datetime


class Article(BaseModel):
    """One retrieved piece of web content. Day 4: snippet-based."""
    url: HttpUrl
    title: str
    snippet: str                         # Tavily's extracted excerpt
    published_date: datetime | None = None
    source_query: str                    # which planner query produced this


class ScoredArticle(BaseModel):
    """Article with Filter's relevance assessment."""
    article: Article
    relevance: int = Field(ge=1, le=5)
    rationale: str                       # one-sentence judge explanation


class Summary(BaseModel):
    """One article summarized, with traceable provenance."""
    article_url: HttpUrl
    summary: str
    key_claims: list[str]                # used by Day 7 citation guardrail


class BriefingState(TypedDict):
    # --- inputs ---
    topic: str                                       # overwrite (set once)
    run_started_at: datetime                         # overwrite

    # --- planner output ---
    search_queries: Annotated[list[str], add]        # append across planner calls

    # --- researcher output ---
    raw_articles: Annotated[list[Article], add]      # append per search

    # --- filter output ---
    scored_articles: Annotated[list[ScoredArticle], add]

    # --- summarizer output ---
    summaries: Annotated[list[Summary], add]

    # --- formatter output ---
    final_briefing: str                              # overwrite (produced once)
```

### Reducer choices, justified

| Key | Reducer | Reasoning |
|---|---|---|
| `topic`, `run_started_at` | overwrite (default) | Set once at graph entry, never updated. |
| `search_queries` | `add` (append) | Planner produces a list; if we ever re-plan (Reflexion-style), new queries extend the old set. Append keeps history auditable in traces. |
| `raw_articles` | `add` (append) | Researcher may return articles across multiple tool calls and multiple queries. Append is the only correct merge — overwrite would drop partial results. Critical for Day 5 parallel fan-out (per-query Researcher invocations writing to the same key). |
| `scored_articles` | `add` (append) | Same reasoning as `raw_articles`; in Day 5's parallel variant, fan-out per article writes scores back to a shared list. |
| `summaries` | `add` (append) | Parallelized per article on Day 5; append is mandatory for fan-in correctness. |
| `final_briefing` | overwrite (default) | Produced once by Formatter. If re-run, last write wins. |

### Error handling philosophy: fail loud

State has no `errors` channel. Non-recoverable errors (malformed URLs from Tavily, API failures, schema validation failures, empty search results) raise exceptions and surface in the LangSmith trace at the exact span where they happened. This is a deliberate early-stage choice: the goal this week is to learn where things actually break, and soft error channels hide that signal.

The Day 7 user-facing error wrapper catches exceptions at the *top level* and returns a structured `BriefingResult` union, but individual nodes do not try to "keep going" after an error. If a node raises, the run fails.

URL validation is a concrete case: `Article` uses Pydantic's strict `HttpUrl`. A malformed URL from Tavily raises a `ValidationError` at the Researcher boundary, which surfaces in the trace with the bad input visible. This is a correctness feature.

### Why not `MessagesState`?

`MessagesState` is the right state when the system is a conversation — one appending stream of `HumanMessage` / `AIMessage` / `ToolMessage`. This system's state is structured artifacts with distinct lifecycles (articles retrieved once, scored once, summarized once, formatted once). Forcing that into a messages list would destroy the type information and make Day 6's eval harness much harder (the eval needs to inspect `scored_articles` directly, not parse them out of message content).

The Researcher sub-agent *does* use `MessagesState` internally — that's the right state for a ReAct loop. The outer graph wraps it and extracts `list[Article]` from the final message, writing structured output back into `BriefingState`. This shape — messages-state inside, custom-state outside — is the canonical pattern for Quadrant 2 systems.

### State lifecycle across a clean run

At graph entry:

```python
{
    "topic": "AI regulation in the EU",
    "run_started_at": datetime.utcnow(),
    "search_queries": [],
    "raw_articles": [],
    "scored_articles": [],
    "summaries": [],
    "final_briefing": "",
}
```

After Planner: `search_queries` has 2–4 entries. Everything else unchanged.

After Researcher: `raw_articles` has ~10–20 entries. Sub-agent's `messages` are not merged upward — they remain inside the Researcher subgraph and appear in LangSmith as child spans, not in outer state.

After Filter: `scored_articles` has the same count as `raw_articles` (every article scored; we drop at Summarizer boundary, not Filter, to keep the trace auditable).

After Summarizer: `summaries` has only high-relevance articles (score ≥ 3).

After Formatter: `final_briefing` is set.

### Checkpointer configuration

`MemorySaver` is attached at graph compile time. Runs are keyed by `thread_id`; each invocation of the pipeline uses a UUID as its thread ID. This gives us:

- Durable state within a run (if a later node crashes, earlier nodes' work is preserved).
- Ability to inspect state at any point via LangSmith.
- Basis for the Day 7 HITL interrupt (resuming from an interrupt needs a checkpointer).

Not used for: multi-turn conversation, cross-run memory. Extension #3 (vector store) is where cross-run memory would live.

---

## 3. Node Specifications

The pipeline has five nodes. Four are plain LLM calls with Pydantic-validated structured output; one (the Researcher) is a `create_agent`-based ReAct subgraph wrapped as a node. Each section below specifies the node's contract — what it reads, what it writes, how it's implemented, how it fails.

### 3.1 Planner

**Responsibility:** Decompose the user's topic into 2–4 focused web search queries that together give the Researcher broad coverage of the topic without redundancy. Single LLM call with structured output.

**Reads from state:** `topic`, `run_started_at`
**Writes to state:** `search_queries` (append via reducer)

#### Why this node exists as a separate step

The Day 2 raw-agent experiment surfaced a concrete bug: given "latest news on AI regulation," the model generated the search query `"latest news AI regulation 2024"`. It hardcoded `2024` on its own, unprompted, in April 2026. The stale-year injection was invisible to the user and actively hurt retrieval quality.

Two fixes were available: (1) instruct the Researcher via system prompt to avoid hardcoded years, or (2) move query planning to a separate node where the current date can be injected and the query formulation is explicit and inspectable. Option 2 is strictly better — it makes the queries a first-class state field (auditable in the trace, reusable across runs, testable independently) and separates "decide what to search" from "execute the searches." The Planner is the payoff for that architectural choice.

A second reason: query diversity. A single Researcher invocation prompted with "research X" tends to run near-duplicate searches. An explicit Planner with an instruction like "produce 2–4 queries that cover distinct angles" forces the model to think about coverage before the Researcher thinks about execution.

#### Implementation: single LLM call with structured output

```python
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from src.schemas import BriefingState


class PlannerOutput(BaseModel):
    queries: list[str] = Field(min_length=2, max_length=4)
    rationale: str  # one-sentence justification of the chosen angles


_planner_model = ChatAnthropic(
    model="claude-sonnet-4-5",
    temperature=0.3,  # slight variation helps query diversity
).with_structured_output(PlannerOutput)


def planner_node(state: BriefingState) -> dict:
    """Decompose topic into 2-4 focused search queries."""
    current_date_iso = state["run_started_at"].date().isoformat()
    result: PlannerOutput = _planner_model.invoke([
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT.format(
            current_date_iso=current_date_iso
        )},
        {"role": "user", "content": f"Topic: {state['topic']}"},
    ])
    return {"search_queries": result.queries}
```

`with_structured_output` is the LangChain wrapper that hands Pydantic validation and schema instruction to the model in one call. No manual JSON parsing, no retry logic needed at this layer — if the schema fails, the call raises.

Note: `rationale` is captured in the Pydantic output but deliberately *not* written to state. It exists for the LangSmith trace (visible in the model's output span) to give Day 6 trajectory evaluation something to inspect, without bloating outer state. This is a general pattern — keep explanatory fields on the model output, keep state lean.

#### System prompt

```
You are a research query planner. Given a topic, produce 2-4 focused web
search queries that together cover the topic from distinct angles.

Today's date is {current_date_iso}.

Rules:
- Do not include years in queries unless the user's topic explicitly asks
  about a specific past period. Search engines rank recent content
  automatically; hardcoded years hurt freshness.
- Queries should be 3-8 words each. Short queries return higher-quality
  results than long natural-language questions.
- Each query should target a distinct angle (e.g., policy, industry
  reaction, technical detail, geographic specificity). Do not produce
  near-duplicate queries.
- Prefer neutral phrasing over leading language. "AI regulation criticism"
  is worse than "AI regulation debate" — the first biases retrieval.

Return a JSON object matching the PlannerOutput schema:
  queries: list of 2-4 search query strings
  rationale: one sentence explaining the coverage angles chosen
```

The "no hardcoded years" rule is the Day 2 fix, now explicit and inspectable.

The "3–8 words" rule reflects how keyword-weighted retrieval (BM25 and successors, which underpin most web search APIs including Tavily) weights rare terms as signal and common words as noise. Short queries concentrate signal; long colloquial queries dilute it. This guidance would flip for pure-vector retrievers, where longer queries can help — a reason to revisit if we ever change retrieval backend.

#### Failure modes

| Failure | Surface | Mitigation |
|---|---|---|
| Fewer than 2 or more than 4 queries | `pydantic.ValidationError` (Field constraints) | Strict validation is the feature. Retry handled at middleware layer if added on Day 5+. |
| Model ignores date injection, emits year in query | Not caught at schema level; visible in trace as a query string containing a 4-digit year | Day 6 trajectory eval: regex-check queries for hardcoded years; mark as trajectory failure. |
| Near-duplicate queries ("EU AI Act" + "European Union AI regulation") | Not caught at schema level; would show as redundant articles later | Day 6 eval can detect by embedding similarity of queries; not a Day 4 concern. |
| Topic is too vague to plan against (e.g., "news") | Model produces overly generic queries | Day 7 input guardrail: reject topic strings shorter than 15 characters. |

#### Trace shape (expected)

```
planner_node (span, ~1-3s)
└── ChatAnthropic (LLM call)
    ├── input: system prompt + user topic
    └── output: PlannerOutput (structured)
```

Single span. If this node takes more than a few seconds, something is wrong.

#### Day 4 stub behavior

```python
def planner_node_stub(state: BriefingState) -> dict:
    return {"search_queries": [
        f"{state['topic']} overview",
        f"{state['topic']} latest developments",
    ]}
```

### 3.2 Researcher

**Responsibility:** Given the planner's search queries, use web search to retrieve relevant articles and return them as structured `Article` objects. This is the only agentic node in the pipeline — the model decides how many searches to run, when a query is exhausted, and when it has enough material.

**Reads from state:** `topic`, `search_queries`
**Writes to state:** `raw_articles` (append via reducer)

#### Why this node is an agent (and the others aren't)

Every other node has a fixed operation: classify this article, summarize this text, format this list into markdown. The Researcher has a genuinely open control flow question: *for this query, how many searches are enough?* That's model-driven — the answer depends on how good the first search result was, whether follow-up terms are obvious, whether the snippets contradict. That's the Quadrant 3 signature (model chooses the path) and is why this node earns the agent complexity cost while the others don't.

The agent has exactly one tool: `web_search`. That's deliberate. A single-tool ReAct agent is still an agent — the "decision" is how many times to call the tool, with what queries, and when to stop.

#### Structured output approach

**Day 4:** manual JSON-in-final-message parsing (`_extract_articles` does `json.loads` + Pydantic validation). This makes the adapter pattern visible.

**Day 5:** swap to `create_agent`'s `response_format=Article` parameter, which has the agent call a structured-output tool natively instead of emitting JSON as text. The Day 5 commit is the before/after comparison worth pointing to in interviews — the manual version exists to make the adapter pattern visible, the structured-output version exists because it's the current best practice.

#### Max iterations: two caps, defense-in-depth

Prompt-level soft cap of 3 searches ("do not exceed 3 search calls") and framework-level hard cap via `create_agent(..., max_iterations=6)`. The prompt cap is about quality (don't thrash similar queries); the framework cap is about safety (bounded execution regardless of what the model decides to do).

#### Implementation: `create_agent` wrapped as a node (Day 4)

```python
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from src.tools.web_search import web_search
from src.schemas import Article, BriefingState


# Built once at module load; create_agent returns a compiled LangGraph runnable
_researcher_agent = create_agent(
    model=ChatAnthropic(model="claude-sonnet-4-5", temperature=0.0),
    tools=[web_search],
    system_prompt=RESEARCHER_SYSTEM_PROMPT,
    max_iterations=6,  # hard safety cap
)


def researcher_node(state: BriefingState) -> dict:
    """Outer-graph node that invokes the researcher sub-agent once per search query."""
    queries = state["search_queries"]
    topic = state["topic"]
    articles: list[Article] = []

    for query in queries:
        result = _researcher_agent.invoke({
            "messages": [{
                "role": "user",
                "content": (
                    f"Topic under investigation: {topic}\n"
                    f"Run web searches to find articles relevant to: {query}\n"
                    f"When you have 3-5 relevant results, return them as a "
                    f"JSON array of objects with keys: url, title, snippet, "
                    f"published_date (ISO 8601 or null). Do not include any "
                    f"commentary outside the JSON."
                )
            }]
        })
        articles.extend(_extract_articles(result, source_query=query))

    return {"raw_articles": articles}
```

This is the adapter. Three things it's doing:

1. **Invoking the sub-agent once per query.** Day 4 is synchronous; Day 5 async variant parallelizes this loop with `asyncio.gather`.
2. **Shaping the outer → inner interface.** Outer state has `search_queries: list[str]` and `topic: str`. The inner agent has `MessagesState` — just a conversation. The adapter translates by building one `HumanMessage` per query containing all the context the inner agent needs.
3. **Shaping the inner → outer interface.** The inner agent's final state is a `MessagesState` with a pile of messages. What we want out of that is `list[Article]`. `_extract_articles` does that parsing.

This inner-MessagesState-outer-BriefingState shape is the canonical Quadrant 2 pattern.

#### The `_extract_articles` helper

```python
import json

def _extract_articles(agent_result: dict, source_query: str) -> list[Article]:
    """Parse the final AIMessage's content as a JSON array of articles.

    Fails loud: any parse error or schema error raises. The trace will show
    the exact agent output that couldn't be parsed.
    """
    final_message = agent_result["messages"][-1]
    payload = json.loads(final_message.content)  # JSONDecodeError if malformed
    return [
        Article(**{**item, "source_query": source_query})
        for item in payload
    ]  # ValidationError if any item is malformed
```

Deliberately minimal. No try/except — if the model returned something that isn't JSON, or returned JSON that doesn't fit `Article`, that's a real failure we want to see in the trace.

#### System prompt

```
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

  {
    "url": str,           // full URL as returned by web_search
    "title": str,         // article title
    "snippet": str,       // 1-3 sentence excerpt, from the search result
    "published_date": str | null  // ISO 8601 date, or null if unknown
  }
```

#### Trace shape (expected)

For each search query, the LangSmith trace shows a subtree rooted at `researcher_node`:

```
researcher_node (span, ~5-15s)
├── researcher_agent (invoke, one per query)
│   ├── model call #1 (AIMessage with tool_use for web_search)
│   ├── web_search tool call
│   ├── model call #2 (AIMessage, either more tool_use or final JSON)
│   ├── (optional) web_search call #2
│   └── model call #3 (AIMessage with final JSON array)
└── _extract_articles (invisible — plain Python, no span)
```

Reading this trace is how Day 6's trajectory eval decides whether the agent "did the right thing." Specifically:

- **Healthy:** 1–3 tool calls per query, final message parses cleanly, articles are non-empty.
- **Unhealthy — thrashing:** 3 tool calls all on near-identical queries (the cap fires but quality is poor).
- **Unhealthy — short-circuit:** 0 tool calls, model tries to answer from training data.
- **Unhealthy — format drift:** final message contains JSON *inside* markdown fencing or with commentary, `_extract_articles` raises, run fails.

#### Failure modes

| Failure | Surface | Day 6/7 mitigation |
|---|---|---|
| Tavily returns 0 results | Parser returns `[]`; downstream nodes see empty `raw_articles` | Day 7 guardrail: raise if `raw_articles` is empty after Researcher |
| Model returns non-JSON | `json.JSONDecodeError` in `_extract_articles` | Day 5+: swap to `response_format` for structured output |
| Model returns JSON with bad schema (missing `url`, invalid URL format) | `pydantic.ValidationError` in `Article` construction | Strict validation is the feature |
| Agent hits `max_iterations` (safety cap) | `create_agent` raises `GraphRecursionError` | Top-level wrapper catches and returns `Failure(reason="max_iterations")` |
| Model hallucinates URLs or snippets | Invisible at this layer; Day 6 faithfulness eval catches it | Day 6 LLM-as-judge; Day 7 citation-verification guardrail |
| Prompt injection via search result content | Agent may follow injected instructions or quote them downstream | Day 2 notes describe this; schema-constrained outputs limit laundering |

#### Day 4 stub behavior

```python
def researcher_node_stub(state: BriefingState) -> dict:
    return {"raw_articles": [
        Article(
            url="https://example.com/ai-regulation-stub",
            title="Stub: EU AI Act Update",
            snippet="This is a stub article for scaffold testing.",
            published_date=None,
            source_query=state["search_queries"][0] if state["search_queries"] else "stub",
        )
    ]}
```

### 3.3 Filter

**Responsibility:** Score each retrieved article's relevance to the topic on a 1–5 scale with a brief rationale. Does not drop articles — scoring is non-destructive, so the trace shows what the Filter judged of everything. The Summarizer boundary is where low-scoring articles are excluded from downstream work.

**Reads from state:** `topic`, `raw_articles`
**Writes to state:** `scored_articles` (append via reducer)

#### Why score-without-dropping

Three reasons this node preserves everything rather than filtering in place:

1. **Auditability.** Day 6 eval can compare the Filter's scores against a human-labeled gold set to measure precision/recall. If the Filter dropped articles silently, the eval couldn't see false rejections.
2. **Trajectory debugging.** When an interviewer asks "show me a failure," having a trace that shows "here's the article the Filter rated 2/5 and dropped; here's why that was wrong" is a stronger story than "here's an article that's not in the output."
3. **Reducer correctness under fan-out.** On Day 5, the Filter will be parallelized per article (one agent invocation per article). Each invocation appends exactly one `ScoredArticle` to state. If we tried to filter-in-place, parallel invocations would need to agree on what to drop, which is a coordination problem we can avoid by keeping everything.

#### Implementation: one LLM call per article

```python
from src.schemas import BriefingState, ScoredArticle, Article
from pydantic import BaseModel, Field


class FilterJudgment(BaseModel):
    relevance: int = Field(ge=1, le=5)
    rationale: str  # one sentence, surfaces in state


_filter_model = ChatAnthropic(
    model="claude-haiku-4-5",  # cheap model; scoring is a simple task
    temperature=0.0,
).with_structured_output(FilterJudgment)


def filter_node(state: BriefingState) -> dict:
    """Score each raw article for relevance. Preserves all articles."""
    topic = state["topic"]
    scored: list[ScoredArticle] = []
    for article in state["raw_articles"]:
        judgment: FilterJudgment = _filter_model.invoke([
            {"role": "system", "content": FILTER_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Topic: {topic}\n\n"
                f"Article title: {article.title}\n"
                f"Article snippet: {article.snippet}\n"
                f"Published: {article.published_date or 'unknown'}\n\n"
                f"Rate relevance 1-5."
            )},
        ])
        scored.append(ScoredArticle(
            article=article,
            relevance=judgment.relevance,
            rationale=judgment.rationale,
        ))
    return {"scored_articles": scored}
```

Choice of `claude-haiku-4-5` is deliberate: scoring an article's relevance against a topic is a simple classification task that doesn't need the frontier model's reasoning depth. Using a cheaper model here keeps per-run cost reasonable when we have 10–20 articles to score.

#### System prompt

```
You are a relevance judge. Given a topic and an article's metadata,
rate how relevant the article is to the topic on a 1-5 scale.

Scale:
  5 = Directly addresses the topic with substantive information.
  4 = Addresses the topic but may be tangential, dated, or thin.
  3 = Mentions the topic but is primarily about something else.
  2 = Only loosely related; shared keywords but different substance.
  1 = Unrelated to the topic despite superficial keyword match.

Judge only on relevance — not on source quality, writing quality, or
factual accuracy. Those are other concerns.

Return a JSON object matching the FilterJudgment schema:
  relevance: integer 1-5
  rationale: one sentence (<=20 words) explaining the score
```

The "judge only on relevance" line is load-bearing. Without it, the model conflates credibility with relevance and produces weird scores.

#### Failure modes

| Failure | Surface | Mitigation |
|---|---|---|
| Model returns `relevance` outside 1–5 | `pydantic.ValidationError` | Strict validation is the feature. |
| Model ignores rubric, conflates relevance with quality | Not caught; would show as poorly-calibrated scores | Day 6 eval: compare Filter scores to human gold labels |
| Positional or verbosity bias (judge prefers longer snippets) | Not caught at this layer | Documented LLM-as-judge risk; addressed in Day 6 Q12 answer |
| Filter drops too many → Summarizer gets nothing | Summarizer raises if `summaries` would be empty | Day 7 guardrail: raise if zero articles score ≥ 3 |

#### Day 4 stub behavior

```python
def filter_node_stub(state: BriefingState) -> dict:
    return {"scored_articles": [
        ScoredArticle(article=a, relevance=4, rationale="stub")
        for a in state["raw_articles"]
    ]}
```

### 3.4 Summarizer

**Responsibility:** For each article scoring ≥ 3 on relevance, produce a 2–3 sentence summary capturing the key claims. Summaries carry URL provenance and an explicit list of key claims for Day 7's citation-verification guardrail.

**Reads from state:** `scored_articles`
**Writes to state:** `summaries` (append via reducer)

#### Relevance threshold is the filter boundary

This is where articles are actually excluded. Articles scoring < 3 are not summarized. The threshold is a constant in `src/config.py`, not a per-run parameter, so it's auditable and consistently applied. `3` is the cutoff because the rubric defines 3 as "mentions the topic but primarily about something else" — below that is not worth the summary cost.

#### Implementation

```python
from src.schemas import BriefingState, Summary
from src.config import RELEVANCE_THRESHOLD  # = 3


class SummaryOutput(BaseModel):
    summary: str = Field(min_length=50, max_length=500)
    key_claims: list[str] = Field(min_length=1, max_length=5)


_summarizer_model = ChatAnthropic(
    model="claude-sonnet-4-5",  # back to frontier model; summary quality matters
    temperature=0.0,
).with_structured_output(SummaryOutput)


def summarizer_node(state: BriefingState) -> dict:
    """Summarize only articles with relevance >= threshold."""
    eligible = [
        sa for sa in state["scored_articles"]
        if sa.relevance >= RELEVANCE_THRESHOLD
    ]
    summaries: list[Summary] = []
    for sa in eligible:
        output: SummaryOutput = _summarizer_model.invoke([
            {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Article title: {sa.article.title}\n"
                f"Source URL: {sa.article.url}\n"
                f"Snippet: {sa.article.snippet}\n\n"
                f"Produce a 2-3 sentence summary and list 1-5 key claims."
            )},
        ])
        summaries.append(Summary(
            article_url=sa.article.url,
            summary=output.summary,
            key_claims=output.key_claims,
        ))
    return {"summaries": summaries}
```

#### System prompt

```
You are a news summarizer. Given an article's title, URL, and snippet,
produce a concise 2-3 sentence summary and an explicit list of the key
claims the article makes.

Critical rules:
- Every claim in the summary and in key_claims MUST be supported by
  content in the snippet. Do not add context from your training data.
- Do not speculate about what the full article might say — you have only
  the snippet.
- If the snippet is too thin to summarize faithfully, return a single
  key_claim: "Snippet insufficient for detailed summary."
- Write in neutral, factual tone. No editorializing.

Return a JSON object matching the SummaryOutput schema:
  summary: 2-3 sentences (50-500 chars) summarizing the snippet
  key_claims: 1-5 specific claims, each a standalone sentence
```

`key_claims` is the hook for Day 7's citation-verification guardrail: every claim in the final briefing must map back to a `key_claim` in exactly one `Summary`, and that `Summary`'s URL is the citation.

#### Failure modes

| Failure | Surface | Mitigation |
|---|---|---|
| Summary hallucinates claims not in snippet | Not caught at this layer | Day 6 faithfulness LLM-as-judge; Day 7 citation guardrail |
| Summary violates length constraints | `pydantic.ValidationError` | Strict validation; consider relaxing `min_length=50` on Day 5 if it trips too often |
| Empty `summaries` (no articles scored ≥ 3) | Run continues to Formatter with empty summaries | Day 7 guardrail: raise before Formatter if `summaries` is empty |
| Snippet is truly insufficient | Model follows escape-hatch instruction | Escape hatch is the feature — preserves faithfulness at cost of coverage |

#### Day 4 stub behavior

```python
def summarizer_node_stub(state: BriefingState) -> dict:
    return {"summaries": [
        Summary(
            article_url=sa.article.url,
            summary=f"Stub summary for: {sa.article.title}",
            key_claims=["Stub claim 1", "Stub claim 2"],
        )
        for sa in state["scored_articles"]
        if sa.relevance >= 3
    ]}
```

### 3.5 Formatter

**Responsibility:** Assemble the summaries into a final markdown briefing with structured sections, inline citations, and a sources list. This is the only node whose output is user-facing.

**Reads from state:** `topic`, `run_started_at`, `summaries`
**Writes to state:** `final_briefing` (overwrite)

#### Why the Formatter is still an LLM call, not a template

A plain Jinja template could produce valid markdown from the summaries without any LLM involvement. The reason to use an LLM here: a good briefing isn't just a list of summaries — it's a synthesized document with grouped themes, a short lead paragraph, and natural transitions between related stories. That requires reading across summaries, which is what an LLM does well and a template cannot.

The tradeoff is that this call is now a faithfulness risk — the Formatter could introduce claims not in the source summaries. Two mitigations: (1) the prompt is explicit that every factual claim must trace to a source URL, (2) Day 7's citation guardrail post-hoc verifies this by matching claims to `key_claims` lists.

#### Implementation: single LLM call, markdown output

```python
from src.schemas import BriefingState


_formatter_model = ChatAnthropic(
    model="claude-sonnet-4-5",
    temperature=0.2,  # slight variation for natural prose
)


def formatter_node(state: BriefingState) -> dict:
    """Assemble summaries into a final markdown briefing."""
    topic = state["topic"]
    date_str = state["run_started_at"].date().isoformat()
    summaries_block = "\n\n".join(
        f"- **Source:** {s.article_url}\n"
        f"  **Summary:** {s.summary}\n"
        f"  **Key claims:** {'; '.join(s.key_claims)}"
        for s in state["summaries"]
    )
    response = _formatter_model.invoke([
        {"role": "system", "content": FORMATTER_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Topic: {topic}\n"
            f"Date: {date_str}\n\n"
            f"Summaries to synthesize:\n\n{summaries_block}"
        )},
    ])
    return {"final_briefing": response.content}
```

No structured output here — the contract is "produce valid markdown," which isn't Pydantic-expressible. Validation is done as a guardrail on Day 7.

#### System prompt

```
You are a news briefing writer. Given a topic and a set of article
summaries, produce a concise markdown briefing.

Structure:
  # {topic} — Briefing for {date}

  A 2-3 sentence lead paragraph synthesizing the overall picture.

  ## Key developments

  2-4 short subsections, each covering a distinct theme drawn from the
  summaries. Each subsection is 2-4 sentences. Use inline citations in
  the form [source](url) after each factual claim.

  ## Sources

  Bulleted list of all URLs used, deduplicated.

Critical rules:
- Every factual claim in the briefing must have an inline [source](url)
  citation pointing to one of the provided source URLs.
- Do not introduce claims not present in the provided summaries.
- Do not editorialize or speculate about implications.
- If summaries conflict, note the disagreement rather than picking a side.

Output only the markdown. No preamble, no trailing commentary.
```

#### Failure modes

| Failure | Surface | Mitigation |
|---|---|---|
| Citations reference URLs not in summaries | Not caught at this layer | Day 7 citation guardrail: parse markdown links, verify all URLs appear in `summaries[*].article_url` |
| Claims in briefing not backed by any `key_claim` | Not caught at this layer | Day 6 faithfulness judge, Day 7 citation guardrail |
| Output is not valid markdown | Not caught at this layer | Day 7 guardrail: parse with a markdown parser, raise on failure |
| Output includes preamble ("Here is the briefing: ...") | Not caught; pollutes output | Mitigated by explicit prompt instruction; could add strip-preamble postprocess if it becomes an issue |

#### Day 4 stub behavior

```python
def formatter_node_stub(state: BriefingState) -> dict:
    lines = [
        f"# {state['topic']} — Stub Briefing",
        "",
        "Stub briefing with the following sources:",
        "",
    ]
    lines.extend(f"- {s.article_url}: {s.summary}" for s in state["summaries"])
    return {"final_briefing": "\n".join(lines)}
```

---

## 4. Graph Topology & Edges

The outer graph is a linear five-node pipeline with one embedded agentic subgraph (the Researcher). No conditional edges, no loops, no fan-out at the outer layer. This is deliberate — it's the minimum complexity that exercises a real Quadrant 2 pattern.

### Topology

```
                                 ┌─────────────┐
   START ─────────────────────►  │   planner   │
                                 └──────┬──────┘
                                        │
                                        ▼
                                 ┌─────────────┐
                                 │ researcher  │ ← contains
                                 └──────┬──────┘   create_agent subgraph
                                        │
                                        ▼
                                 ┌─────────────┐
                                 │   filter    │
                                 └──────┬──────┘
                                        │
                                        ▼
                                 ┌─────────────┐
                                 │ summarizer  │
                                 └──────┬──────┘
                                        │
                                        ▼
                                 ┌─────────────┐
                                 │  formatter  │
                                 └──────┬──────┘
                                        │
                                        ▼
                                       END
```

### Edge specification

All edges in the Day 4 graph are **plain (static)** — no conditional edges. The pipeline runs top to bottom on every invocation.

| From | To | Type | Condition |
|---|---|---|---|
| `START` | `planner` | plain | always |
| `planner` | `researcher` | plain | always |
| `researcher` | `filter` | plain | always |
| `filter` | `summarizer` | plain | always |
| `summarizer` | `formatter` | plain | always |
| `formatter` | `END` | plain | always |

### Why no conditional edges at the outer layer

A conditional edge would imply the outer graph has a decision to make. On Day 4 it doesn't — every topic goes through every node in the same order. The *only* place model-driven routing exists is inside the Researcher's `create_agent` subgraph, where the inner graph's conditional edge decides "loop back to model for another tool call" vs. "go to END and return JSON." That's the agentic part of the system. Everything outside it is workflow.

**This is the concrete payoff of the workflow-vs-agent framing from Days 1 and 3:** when you look at this graph, four nodes are workflow, one node contains an agent. The system is Quadrant 2 — a workflow with an embedded agent — and the topology makes that visually obvious.

Candidates for conditional edges in future iterations:

- **`researcher → filter` vs. `researcher → planner`** (re-plan loop). If the Researcher returns too few articles, route back to the Planner for query refinement. This is a Reflexion-style pattern; scoped out of this week but a natural extension.
- **`filter → summarizer` vs. `filter → END`** (early termination). If the Filter scores every article below threshold, terminate with a "no relevant coverage found" message rather than invoking Summarizer on an empty set.
- **`formatter → END` vs. `formatter → human_review`** (HITL interrupt). Day 7's HITL checkpoint.

Each of these is a one-line change in graph topology (`add_conditional_edges` instead of `add_edge`) — the architecture supports them without refactor.

### Fan-out / parallelism (where it will live)

Day 4 is fully sequential. Day 5's async variant introduces parallelism at two layers:

1. **Inside the Researcher node** — the per-query `for` loop becomes `asyncio.gather`, running one `_researcher_agent.invoke(...)` per planner query concurrently. All invocations append to `raw_articles`, which the `add` reducer merges deterministically.
2. **Inside the Filter node** — the per-article `for` loop becomes `asyncio.gather`, running N concurrent relevance judgments.

These are *intra-node* parallelism. The outer graph topology is unchanged — the nodes themselves run sequentially relative to each other, but each node internally does concurrent work.

- **Intra-node parallelism** (Day 5): the node decides to parallelize its internal work. No change to graph shape. Fan-out lives inside the node function.
- **Inter-node parallelism** (Day 5 multi-agent variant): the graph has explicit parallel branches. A node returns edges to multiple downstream nodes, LangGraph fans them out, a reducer merges state on rejoin.

Inter-node parallelism is what Q3 is actually about ("how do you control sequential vs parallel execution in multi-agent systems"). Intra-node parallelism is a lower-level optimization that doesn't change the graph's logical shape.

### Graph construction code

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.schemas import BriefingState
from src.nodes.planner import planner_node
from src.nodes.researcher import researcher_node
from src.nodes.filter import filter_node
from src.nodes.summarizer import summarizer_node
from src.nodes.formatter import formatter_node


def build_graph(checkpointer=None):
    """Build and compile the briefing pipeline."""
    graph = StateGraph(BriefingState)

    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("filter", filter_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("formatter", formatter_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "filter")
    graph.add_edge("filter", "summarizer")
    graph.add_edge("summarizer", "formatter")
    graph.add_edge("formatter", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())
```

Ten lines of topology. The actual engineering is in the node implementations.

### Graph construction during scaffold phase

Day 4 morning: `build_graph` is called with stub node functions. The graph compiles, runs end-to-end, and produces a stub briefing. Proves the topology works. Only then do we replace node implementations one at a time.

The discipline — scaffold first, fill in nodes one at a time — means a broken node is always the most recent change.

### Thread IDs and invocation

Each invocation gets a fresh UUID `thread_id`. Checkpointer stores state keyed by thread. LangSmith traces are also grouped by thread, making a trace searchable by its run's UUID. If we ever need to resume an interrupted run (Day 7 HITL), the thread_id is how we look it up.

---

## 5. I/O Contract

The graph itself is not the user-facing surface. A top-level function `run_briefing(topic: str) -> BriefingResult` wraps the compiled graph, handles invocation setup, catches exceptions, and returns a structured result. This is the boundary where "internal system that fails loud" meets "user-facing interface that returns structured outcomes."

### Input

Single string: the topic the user wants briefed. Validation happens at the top-level wrapper, not inside the graph:

- Non-empty after stripping whitespace.
- Minimum length of 15 characters.
- Maximum length of 500 characters.
- Must be a `str`.

Validation failures raise `ValueError` before the graph is invoked.

### Output

A `BriefingResult` discriminated union — either a success carrying the briefing, or a failure carrying a structured reason.

```python
from pydantic import BaseModel
from typing import Literal, Union
from datetime import datetime


class BriefingSuccess(BaseModel):
    status: Literal["success"] = "success"
    topic: str
    briefing_markdown: str
    run_started_at: datetime
    run_completed_at: datetime
    thread_id: str               # for LangSmith trace lookup
    source_urls: list[str]       # flattened from summaries


class BriefingFailure(BaseModel):
    status: Literal["failure"] = "failure"
    topic: str
    reason: Literal[
        "invalid_topic",         # input validation failed
        "no_search_results",     # Researcher returned empty
        "no_relevant_articles",  # Filter scored everything below threshold
        "max_iterations",        # Researcher agent hit its safety cap
        "schema_violation",      # a node's structured output failed validation
        "api_error",             # upstream LLM or Tavily API failure
        "unknown",               # catch-all
    ]
    message: str                 # human-readable, for UI/logging
    run_started_at: datetime
    run_failed_at: datetime
    thread_id: str | None        # None if failure was before graph invocation


BriefingResult = Union[BriefingSuccess, BriefingFailure]
```

The `reason` field is a closed enum. Downstream code can pattern-match on `reason` without parsing messages. `message` is free-form for human consumption; `reason` is the machine-readable channel.

`thread_id` on both variants makes every result traceable to its LangSmith run.

### Top-level wrapper implementation

```python
from uuid import uuid4
from datetime import datetime
from pydantic import ValidationError
from langgraph.errors import GraphRecursionError
from src.graph import build_graph
from src.schemas import BriefingSuccess, BriefingFailure, BriefingResult


_MIN_TOPIC_LEN = 15
_MAX_TOPIC_LEN = 500
_compiled_graph = build_graph()  # built once at module load


def run_briefing(topic: str) -> BriefingResult:
    """Run the briefing pipeline for one topic. Never raises; always returns a result."""
    started_at = datetime.utcnow()

    # --- input validation ---
    if not isinstance(topic, str):
        return BriefingFailure(
            topic=str(topic), reason="invalid_topic",
            message=f"Expected str, got {type(topic).__name__}.",
            run_started_at=started_at, run_failed_at=datetime.utcnow(),
            thread_id=None,
        )
    topic = topic.strip()
    if not (_MIN_TOPIC_LEN <= len(topic) <= _MAX_TOPIC_LEN):
        return BriefingFailure(
            topic=topic, reason="invalid_topic",
            message=f"Topic must be {_MIN_TOPIC_LEN}-{_MAX_TOPIC_LEN} chars; got {len(topic)}.",
            run_started_at=started_at, run_failed_at=datetime.utcnow(),
            thread_id=None,
        )

    # --- graph invocation ---
    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "topic": topic,
        "run_started_at": started_at,
        "search_queries": [],
        "raw_articles": [],
        "scored_articles": [],
        "summaries": [],
        "final_briefing": "",
    }

    try:
        final_state = _compiled_graph.invoke(initial_state, config=config)
    except GraphRecursionError as e:
        return BriefingFailure(
            topic=topic, reason="max_iterations",
            message=f"Agent exceeded max iterations: {e}",
            run_started_at=started_at, run_failed_at=datetime.utcnow(),
            thread_id=thread_id,
        )
    except ValidationError as e:
        return BriefingFailure(
            topic=topic, reason="schema_violation",
            message=f"A node produced output that failed validation: {e}",
            run_started_at=started_at, run_failed_at=datetime.utcnow(),
            thread_id=thread_id,
        )
    except Exception as e:
        # Last-resort catch. Everything else is a structured failure above.
        return BriefingFailure(
            topic=topic, reason="unknown",
            message=f"{type(e).__name__}: {e}",
            run_started_at=started_at, run_failed_at=datetime.utcnow(),
            thread_id=thread_id,
        )

    # --- success path ---
    return BriefingSuccess(
        topic=topic,
        briefing_markdown=final_state["final_briefing"],
        run_started_at=started_at,
        run_completed_at=datetime.utcnow(),
        thread_id=thread_id,
        source_urls=[str(s.article_url) for s in final_state["summaries"]],
    )
```

### Why this boundary exists

Two separate design goals collide at the wrapper:

1. **Internal: fail loud.** Every node raises on schema violations, parse errors, validation failures.
2. **External: never raise.** Calling code should never see a stack trace. Failures come back as typed data.

The wrapper is the translation layer. Exceptions inside the graph hit the wrapper's `try/except`, get classified into a `BriefingFailure.reason`, and are returned as data. `thread_id` is preserved so the raw exception is still recoverable via the LangSmith trace.

### Exception handling: interim posture

The broad `except Exception` with `reason="unknown"` is a deliberate interim choice. The wrapper's external contract ("never raises") requires some last-resort catch, but catching broadly also hides bugs during iteration. The plan: let the Day 6 eval harness run the pipeline against 10–15 topics and collect the actual exception types that fire in practice. Recurring failure modes then get promoted to named `reason` enum values, shrinking the `unknown` bucket over time. A stronger long-term pattern (not built this week) would gate the broad catch behind an environment flag so `unknown` re-raises in development but is caught in production — giving iteration-time visibility without sacrificing the production contract. Tracked as a post-Day-7 refinement.

### What this is not

- **Not a retry layer.** `run_briefing` runs the graph exactly once.
- **Not a cache.** Same topic called twice runs twice.
- **Not async.** Day 5 introduces an `async def arun_briefing` alongside this; the sync version stays for simple CLI use and for the eval harness.
- **Not concurrent.** A single `run_briefing` call is one pipeline invocation.

### CLI invocation

```python
# src/cli.py
import sys
from src.runner import run_briefing


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.cli <topic>")
        sys.exit(1)
    result = run_briefing(" ".join(sys.argv[1:]))
    if result.status == "success":
        print(result.briefing_markdown)
        print(f"\n---\nThread: {result.thread_id}")
    else:
        print(f"FAILURE ({result.reason}): {result.message}", file=sys.stderr)
        if result.thread_id:
            print(f"Trace: {result.thread_id}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
```

---

## 6. Human-in-the-Loop (Day 7 Sketch)

Day 4 has no HITL. This section documents where the Day 7 interrupt will be inserted and why — so the state schema and graph topology from Sections 2 and 4 already support it without refactor.

### The interrupt point

Between Summarizer and Formatter. The interrupt pauses graph execution after all article summaries are produced but before the final briefing is assembled. A human reviews the `summaries` list and chooses:

- **Approve** — resume, Formatter produces the briefing from `summaries` as-is.
- **Edit** — human modifies one or more summaries, then resumes.
- **Reject with feedback** — human writes a short instruction; the graph re-runs Summarizer on flagged articles with the feedback injected.

These are the three canonical HITL decision types from Q11: approve/reject, edit, and review-with-feedback.

### Why this is the right interrupt point

1. **Highest leverage per minute of human time.** The Researcher runs too many searches to review; the Filter's scoring is mechanical; the Formatter's output is derivative. The Summarizer is the first node whose output is small enough to review (typically 5–10 summaries) and whose correctness is critical — if a summary hallucinates, the citation in the final briefing will look supported but be false.
2. **It's where hallucination lives.** The Summarizer's `key_claims` field is the explicit hook for human verification.
3. **Editing is cheap at this stage.** An edited `Summary` is valid input to the Formatter without any other change.

### Implementation sketch (Day 7)

The graph topology changes minimally: the `summarizer → formatter` plain edge becomes a conditional edge routing to either `formatter` (approve path) or a new `human_review` node (interrupt path). `interrupt` from `langgraph.types` pauses the graph; `Command(resume=...)` resumes it.

The `MemorySaver` checkpointer from Day 3 is what makes this work — state is already persisted per thread.

### What the interrupt does not do

- Not approval for every tool call (would be too chatty at 10+ searches per run).
- Not approval before writing anything to external systems — this pipeline has no external writes.

---

## 7. Reliability & Error Handling

### Retry policy

**Retry at the tool boundary, not the node boundary.** Using `tenacity` on the `web_search` tool function:

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
    reraise=True,
)
def _tavily_search(query: str) -> list[dict]:
    ...
```

Key choices:

- **3 attempts max.** One retry usually covers transient issues; two additional cover brief outages.
- **Exponential backoff starting at 1s, capped at 8s.** Standard pattern; doesn't hammer struggling upstream.
- **Only retry transient exceptions.** Timeouts and 5xx are transient; 4xx and malformed-response errors are not.
- **`reraise=True`.** After three failures, the original exception propagates to the wrapper.

LLM calls (Anthropic SDK) have their own built-in retry behavior at the SDK level.

### Rate limiting

Three caps, at three different layers:

1. **Per-run Tavily cap.** No more than `MAX_SEARCH_CALLS_PER_RUN = 6` per invocation. Day 4 relies on the prompt-level cap; explicit enforcement added Day 5.
2. **Researcher `max_iterations`.** `create_agent(..., max_iterations=6)`.
3. **Global per-minute LLM rate.** Not enforced in application code — relies on 429 responses + retry policy.

The three caps protect against three different failure modes:

- Prompt-cap: quality concern (don't thrash similar queries).
- `max_iterations`: safety concern (bounded execution).
- Per-run Tavily cap: budget concern.

### Timeouts

| Call | Timeout | Rationale |
|---|---|---|
| Tavily search | 10s | Search should be fast. |
| LLM calls | 60s (SDK default) | Summarization can take 20–30s under load. |
| Full `run_briefing` | no hard timeout | Natural ceiling is sum of bounded steps. |

### What is explicitly not implemented

- **Dead-letter queue.** Failures return as `BriefingFailure` and are logged; not stored for replay.
- **Alerting.** No Slack / PagerDuty / email on failure.
- **Automatic recovery.** Max-iteration failures don't trigger re-runs.
- **Cost metering.** Per-run token cost is visible in LangSmith but not enforced as a hard cap.

These are production concerns, deliberately deferred.

---

## 8. Testing & Day 4 Acceptance Criteria

### Day 4 acceptance criteria

The Day 4 scaffold is complete when all of the following hold:

1. **Repo builds.** `pip install -e .` succeeds in a clean virtualenv. Required env vars (`ANTHROPIC_API_KEY`, `TAVILY_API_KEY`, `LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`) are documented in `README.md` and the project fails fast with a clear message if they're missing.
2. **Stub graph runs end-to-end.** `python -m src.cli "EU AI Act enforcement"` with all five nodes as stubs produces a stub briefing and exits 0. This proves the topology, not any node's logic.
3. **Real Researcher node is wired in.** The other four nodes remain stubs; the Researcher is a real `create_agent` invocation with `web_search` as its tool.
4. **One real end-to-end run succeeds.** On one topic, the real Researcher retrieves ≥ 3 articles, downstream stubs pass them through, and the final stub briefing contains at least one real URL.
5. **LangSmith trace is complete.** The trace shows, in order: planner_node → researcher_node (with nested per-query agent subgraphs) → filter_node → summarizer_node → formatter_node → END.
6. **`run_briefing` returns a typed result on success and failure.** Trip the topic-length validator (e.g., `run_briefing("AI")`) to produce a `BriefingFailure(reason="invalid_topic")`.
7. **Scaffold discipline held.** Git log shows: "scaffold with stubs compiling" → "stub graph runs end-to-end" → "Researcher made real".

### What Day 4 does *not* test

- Not tested: Researcher's quality on hard topics, Filter's calibration, Summarizer's faithfulness, Formatter's prose quality. Those are Day 5/6 concerns.
- Not tested: async, parallel fan-out, retries under load. Day 5/7.
- Not tested: guardrails, HITL interrupts, citation verification. Day 7.
- Not tested: the eval harness. Day 6.

### Lightweight unit tests to write during scaffolding

- `test_schemas.py`: `Article` rejects malformed URLs; `ScoredArticle` rejects out-of-range scores; `BriefingFailure.reason` rejects unknown values.
- `test_graph_topology.py`: `build_graph()` compiles. Compiled graph's `get_graph()` has the expected five node names and expected edge set.
- `test_run_briefing_validation.py`: empty string, whitespace-only, too-short, non-string input all return `BriefingFailure(reason="invalid_topic")`.

Not writing: tests that mock the LLM or Tavily. Not needed for a Day 4 scaffold.

### What "real end-to-end run" should produce for inspection

Save the following artifacts after the real Day 4 run:

- `examples/day4_eu_ai_act_trace.md` — markdown dump of the LangSmith trace URL and a screenshot of the trace tree.
- `examples/day4_eu_ai_act_result.json` — serialized `BriefingSuccess` with `source_urls` and `thread_id`.
- A one-paragraph reflection in `day4_summary.md` noting anything surprising about the first real run.

The Day 4 summary is written tomorrow morning before starting Day 5.

## Multi-Agent Variant (Day 5 PM)

### Why
To directly compare a workflow-with-embedded-agent (Quadrant 2) against a
supervisor-pattern multi-agent (Quadrant 4) on identical inputs. The test
is whether the added complexity earns its cost in quality or catches errors
the single-agent version misses.

### Topology
[your sketch, in prose or ASCII]

### The one model-driven branch
Critic(draft)'s accept/revise verdict is the only place in the graph where
control flow depends on model judgment. The supervisor itself is a
deterministic conditional edge — chosen deliberately; an LLM call for
`if verdict == 'revise' and count < 2` would be performance theater.

### Revision bound
Maximum 2 revision rounds. On the third rejection, force-accept the current
draft. This is a behavioral guardrail — it prevents runaway critic/writer
disagreement from burning tokens without progress.

### State additions
[the 4 new fields, with reducer choices and one-line rationale each]

### Known failure modes
- Critic produces malformed verdict → fallback to accept, log.
- Writer ignores feedback on revision → revision_count still advances; eventually force-accepts. (Detectable in eval by comparing revised drafts to their feedback.)
- Critic rejects every draft → force-accept at round 2. Metric worth tracking: what % of runs hit the force-accept path?


## Block1 Final Design
START
  ↓
Researcher              (reuse from single-agent, unchanged)
  ↓
Critic(articles)        (replaces Filter; outputs scored_articles with issue flags)
  ↓
Writer                  (drafts from scored_articles; on revision, sees critic_feedback)
  ↓
Critic(draft)           (outputs verdict + always writes final_briefing = draft)
  ↓
[conditional edge: should_continue_revising]
  ├── accept OR revision_count >= 2  → END
  └── revise AND revision_count < 2  → Writer