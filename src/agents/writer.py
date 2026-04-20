"""
Writer agent — drafts the final briefing markdown from scored articles.

This is the agent the Critic(draft) reviews. It's the "create" half of the
create/critique loop. First-round drafts (revision_count=0) are unconditioned;
revision drafts include the Critic's feedback from the prior round and are
instructed to address those items specifically.

One agent, two modes. Unlike the Critic (two files for clarity), the Writer
is genuinely doing one job in both modes — synthesize a briefing from a
source list. Only the prompt context differs by revision round, so keeping
a single agent with a single prompt template is correct here.

Design notes:
- Structured output (response_format) matches the single-agent Formatter.
  We get a typed briefing_markdown field back plus a small revision_rationale
  field — empty on first draft, populated on revisions. The rationale is
  not stored in state; it's purely for LangSmith trace readability, which
  is the "how do you audit a multi-agent system" interview answer made
  concrete.
- No tools. Pure generation. The Writer doesn't re-query the web or
  re-rank sources — it synthesizes what the upstream agents provided.
  If the Writer can't produce a faithful briefing from the given sources,
  that's a signal to change the upstream (planner/researcher/critic-articles),
  not to give the Writer escape hatches.
- Temperature 0.2 — same as the single-agent Formatter. Enough variance
  for prose quality, low enough for eval reproducibility.
- Citation rules repeated verbatim from the Formatter. The Critic enforces
  these; the Writer has to do them right in the first place or every
  revision round burns cycles fixing citations instead of addressing real
  faithfulness issues.
- Flag awareness: the Writer sees each article's flags and is instructed
  to caveat or downweight flagged articles appropriately. This is how the
  multi-agent variant catches issues the single-agent Filter's binary
  cutoff misses.
"""

from langchain.agents import create_agent
from pydantic import BaseModel, Field

from src.llm import DEFAULT_MODEL, get_chat_model


class _WriterOutput(BaseModel):
    """Structured output for the Writer.

    headline and briefing_markdown together form the final briefing content.
    revision_rationale is for trace-readability only; not stored in state.
    """

    headline: str = Field(
        description=(
            "A short, factual headline for the briefing. No sensationalism."
        )
    )
    briefing_markdown: str = Field(
        description=(
            "The full briefing as markdown. Must start with a 2-3 sentence "
            "executive summary, followed by body sections organized by theme "
            "(not by article). Every factual claim must be followed by a "
            "[source: <url>] citation drawn from the provided article list."
        )
    )
    revision_rationale: str = Field(
        default="",
        description=(
            "If this is a revision draft (revision_count > 0), a one or two "
            "sentence note describing what you changed from the previous "
            "draft and how the Critic's feedback items were addressed. Empty "
            "on first-round drafts. Not consumed downstream — purely for "
            "trace readability."
        ),
    )


WRITER_SYSTEM_PROMPT = """\
You are a news briefing writer. You will be given a topic, a list of scored
and flagged source articles, and — on revision rounds — feedback from a
critic on a previous draft.

Your job is to produce a synthesized briefing markdown document.

Structure requirements:
- Start with a 2-3 sentence executive summary.
- Organize the body by theme, not by article. Multiple articles may
  contribute to the same theme.
- Every factual claim must be followed by a bracketed citation using one
  of the URLs from the article list. Format: [source: <url>]
- Do not introduce claims not supported by the provided articles.
- Do not cite URLs outside the provided article list.
- Keep tone factual and professional. No speculation, no editorializing.

Flag awareness:
- Articles with flag "unsupported_claim" or "low_credibility" should not
  be cited for their central claims without a caveat. If you use them,
  note the limitation in the briefing (e.g., "as reported by [source] —
  though not independently verified").
- Articles flagged "outdated" should be caveated as historical context.
- Articles flagged "duplicate" should be treated as redundant with the
  original — cite one or the other, not both.

Revision handling:
- If the user message includes "CRITIC FEEDBACK", this is a revision draft.
  Read the feedback carefully. Address each item specifically. Do NOT
  rewrite the entire draft from scratch; preserve what was working and
  fix what the Critic flagged.
- On revisions, populate revision_rationale with a one or two sentence
  note on what you changed. On first drafts (no feedback block), leave
  revision_rationale empty.

Output: a structured object with headline, briefing_markdown, and
revision_rationale.
"""


def build_writer_agent():
    """Construct the Writer agent. Called once per graph build.

    Temperature 0.2 for prose-quality variance, matching the single-agent
    Formatter. Day 6 eval sweeps can vary this to compare determinism vs.
    readability.
    """
    return create_agent(
        model=get_chat_model(model=DEFAULT_MODEL, temperature=0.2),
        tools=[],
        system_prompt=WRITER_SYSTEM_PROMPT,
        response_format=_WriterOutput,
    )