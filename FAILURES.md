# FAILURES.md — Observed Failure Modes

Honest log of failure modes observed during development of the news briefing agent. Each entry: what happened, root cause (if known), mitigation status, and interview relevance.

Captured as observed, not reconstructed from memory. Not every entry is a bug — some are "plausible-looking but unverified" cases that motivate design choices downstream.

---

## Researcher: JSON parse failure on malformed LLM output

**Trace:** `f04ed178-0c42-4833-8dd1-d5313a3c50a4`
**Trace URL:** https://smith.langchain.com/o/18018a9f-c029-4d1d-b822-750c9d951724/projects/p/5bab7076-8548-48b6-94e3-2d0a0d161a5e?peek=20260420T004438Z019da858-b4c8-77a0-95ba-ea3379ecf837&peeked_trace=20260420T004438984092Z019da858-b4c8-77a0-95ba-ea3379ecf837
**When:** Day 5 morning, during Summarizer integration.
**Symptom:** `json.JSONDecodeError: Expecting ',' delimiter: line 5 column 337 (char 563)` bubbling up from `researcher_node._extract_articles`. Classified as `reason="unknown"` by `run_briefing`'s last-resort catch.

**Root cause:** Hand-rolled JSON extraction from the research sub-agent's final message. The prompt told the model to emit a raw JSON array; the model complied on most runs but occasionally produced technically-invalid JSON — almost certainly an unescaped double-quote inside an article `title` or `snippet`. `raw.find("[")` to `raw.rfind("]")` handles prose-wrapping but nothing enforces string-escape validity.

**Fix:** Replaced text parsing with `response_format=_ResearcherOutput` on `create_agent`. The sub-agent's final step is now a schema-coerced tool call, not free-form text, so the API enforces validity and `agent.invoke(...)` returns a typed Pydantic instance via `result["structured_response"]`. Same fix later applied proactively to Summarizer (`with_structured_output`) and Formatter.

**Interview relevance:** Concrete example of "moved validation to the API boundary where it's enforced, rather than after the fact where it's brittle." Also a concrete case where a recurring failure classified as `reason="unknown"` was promoted — exactly the evolution path flagged in the runner's docstring.

---

## Formatter: Plausible quote attribution that may not be faithful to source

**Trace:** `fd3f22a8-1e3a-4299-ab0f-43ddd79625b4`
**Topic:** "Fed interest rates decisions 2026"
**When:** Day 5 morning, second end-to-end validation run.
**Symptom:** Briefing contains this sentence under "Geopolitical Factors":

> The Fed's post-meeting statement noted 'uncertain' impacts from a war with Iran [source: https://www.cnbc.com/2026/03/18/fed-interest-rate-decision-march-2026.html].

The attribution frames 'uncertain' as a direct quote from the Fed's post-meeting statement, but the cited source is the CNBC article, not the Fed statement itself. Two possibilities that the briefing alone cannot distinguish:
1. CNBC directly quoted the Fed, the Summarizer faithfully relayed the quote with its attribution, and the Formatter preserved it correctly. This is fine.
2. CNBC paraphrased the Fed, the Summarizer or Formatter introduced quote marks that weren't in the source snippet, and the result is a fabricated attribution. This is a faithfulness failure.

**Root cause:** Unknown without opening the trace and comparing the CNBC article's actual text against the Summarizer's per-article output. The pipeline has no current mechanism to verify that specific claims — especially quoted claims — trace back to the cited source.

**Mitigation status:** Not fixed. Flagged as a Day 6 eval target. The LLM-as-judge faithfulness check is the intended mechanism: "for each bracketed citation, does the preceding claim appear in the summary of the cited article?" Quoted claims warrant a stricter variant: "does this exact string appear in the snippet?"

**Interview relevance:** Real example of why post-hoc faithfulness evaluation matters. The briefing reads well, cites sources, follows the format — and still may contain a subtle fabrication. This is the gap between "looks correct" and "is correct" that motivates the Day 6 eval work. Also a concrete answer to "how would you evaluate an agent whose output is prose?" — you need a faithfulness judge, not just a schema check.

---