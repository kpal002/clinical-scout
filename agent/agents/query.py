"""Query Agent — generates 3 specialised search strategies in parallel."""
from __future__ import annotations

import json
from typing import Dict, List

import anthropic

_SYSTEM = """You are a biomedical librarian expert in PubMed MeSH query formulation.

For the given question, produce 3 SPECIALISED search strategies targeting different
evidence types:
  • "systematic_review" — meta-analyses and systematic reviews (highest evidence)
  • "rct"              — randomized controlled trials and clinical trials
  • "observational"    — cohort studies, case-control, mechanistic research

Return ONLY valid JSON, no other text:
{
  "systematic_review": ["query1", "query2"],
  "rct":               ["query1", "query2"],
  "observational":     ["query1", "query2"]
}"""

_DEBUNKER_SYSTEM = """You are a biomedical librarian. For the given medical myth or claim,
produce 3 SPECIALISED PubMed search strategies:
  • "systematic_review" — meta-analyses and systematic reviews on this topic
  • "rct"              — experimental studies directly testing the claim
  • "observational"    — epidemiological and mechanistic evidence

Return ONLY valid JSON, no other text:
{
  "systematic_review": ["query1", "query2"],
  "rct":               ["query1", "query2"],
  "observational":     ["query1", "query2"]
}"""


def run(
    question: str,
    mode: str,
    client: anthropic.Anthropic,
) -> Dict[str, List[str]]:
    """Ask Claude to produce 3 specialised search strategies.

    Returns a dict mapping strategy name → list of query strings.
    """
    system = _DEBUNKER_SYSTEM if mode == "debunker" else _SYSTEM
    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=768,
        system=system,
        messages=[{"role": "user", "content": question}],
    )
    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    strategies: Dict[str, List[str]] = json.loads(raw)
    # Guarantee the three expected keys exist
    for key in ("systematic_review", "rct", "observational"):
        strategies.setdefault(key, [question])
    return strategies
