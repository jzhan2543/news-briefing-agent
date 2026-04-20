"""
Critic(draft) agent — reviews a Writer draft, returns a verdict.

This is the one genuinely model-driven decision in the multi-agent graph.
Everything else is deterministic routing; the Critic's accept/revise verdict
is what earns the "supervisor pattern" label. The supervisor itself is a
conditional edge (if verdict == "revise" and count < 2); the Critic's
judgment is what the conditional edge routes on.

Design notes:
- response_format with a Literal["accept", "revise"] enum forces the model
  to commit. Without structured output, the model defaults to wishy-washy
  "mostly good but could be better" language that neither accepts nor
  truly rejects. Day 2 JSON-output failures motivated this pattern; same
  fix applies here.
- Feedback is capped at 3 items, each short. Open-ended revision feedback
  lets the Critic write a novel and the Writer overreact. Forcing the
  Critic to prioritize is a quality lever that also bounds cost — both
  Critic's output tokens and the Writer's next-round input tokens.
- Faithfulness > citation discipline > flag resolution > coherence,
  in that order of priority. Communicated explicitly in the prompt because
  models left to their own devices will nitpick tone (which is what
  temperature does) rather than catch hallucinations (which is what we
  hired a Critic for).
- No tools. The Critic reads the draft and the scored articles; it doesn't
  search the web or re-verify externally. That's intentional scope —
  faithfulness *relative to the provided sources* is the contract.
  Independent fact-checking is a Day 6+ extension.
- Temperature 0.0: verdicts should be stable across runs. A draft that's
  accepted on one run and rejected on the next with no changes is a
  Critic calibration failure that shows up in eval.
"""

from typing import Literal

from langchain.agents import create_agent
from pydantic import BaseModel, Field

from src.llm import DEFAULT_MODEL, get_chat_model


class _CriticDraftOutput(BaseModel):
    """Structured output for Critic(draft).

    verdict is a hard enum — the model must commit. feedback_items is
    capped at 3 to force prioritization; on "accept" it can be empty or
    a short positive note. overall_note is optional free text for
    trace-readability, not consumed downstream.
    """

    verdict: Literal["accept", "revise"]
    feedback_items: list[str] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "If verdict is 'revise', up to 3 specific, actionable items the "
            "Writer should fix in the next draft. Each item should be one or "
            "two sentences. If verdict is 'accept', leave empty."
        ),
    )
    overall_note: str = Field(
        default="",
        description=(
            "Optional one-sentence summary of your reasoning, for trace "
            "readability. Not consumed downstream."
        ),
    )


CRITIC_DRAFT_SYSTEM_PROMPT = """\
You are a news briefing critic. You will be given a draft briefing and the
list of scored source articles it was written from. Your job is to decide
whether the draft is acceptable as a final briefing, or whether it needs
revision.

Check these four things, in this order of priority:

1. FAITHFULNESS (most important): Every claim in the draft must be supported
   by at least one of the source articles. Inventions, extrapolations beyond
   what the articles say, and "confident-sounding" filler that isn't in any
   source are all revision-worthy.

2. CITATION DISCIPLINE: Every factual claim should be followed by a
   [source: <url>] citation, and the URL should be one of the URLs in the
   scored_articles list. Bare assertions without citations, or citations
   to URLs not in the provided list, are revision-worthy.

3. FLAG RESOLUTION: If a source article was flagged by the upstream critic
   (flags field in scored_articles), the draft should not treat that article
   as authoritative. Specifically: articles flagged "unsupported_claim" or
   "low_credibility" should not be cited for their central claims without
   a caveat in the draft. Articles flagged "outdated" should be caveated
   as historical context. Articles flagged "duplicate" should not be cited
   redundantly.

4. COHERENCE (least important): The draft should read as a synthesized
   briefing — executive summary, themed sections, integrated sources —
   not as a list of article summaries stitched together. Low-priority
   because prose quality is the Writer's job; don't reject a
   faithful, well-cited draft over stylistic preferences.

Decision rules:
- Revise only if you find a concrete issue in priorities 1-3 above, OR if
  priority 4 is so broken the briefing doesn't read as a briefing at all.
- Do NOT revise for grammar, word choice, or tone. Those are the Writer's
  temperature handling their own job.
- If you do revise, give at most 3 feedback items. Prioritize. If there are
  more than 3 issues, pick the 3 most important and flag them; the Writer
  will regenerate and your next review can catch remaining issues.
- If the draft has zero faithfulness or citation issues, ACCEPT. Do not
  nitpick a working briefing.

Output: a structured verdict (accept | revise) and feedback_items list.
"""


def build_critic_draft_agent():
    """Construct the Critic(draft) agent. Called once per graph build.

    Uses DEFAULT_MODEL at temperature 0.0 — verdicts should be stable
    across runs. Day 6 eval sweep is a natural place to try a different
    judge model if self-preference bias shows up (e.g., a Critic in the
    same model family as the Writer systematically accepting drafts).
    """
    return create_agent(
        model=get_chat_model(model=DEFAULT_MODEL, temperature=0.0),
        tools=[],
        system_prompt=CRITIC_DRAFT_SYSTEM_PROMPT,
        response_format=_CriticDraftOutput,
    )