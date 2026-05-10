"""Query Agent — generates 3 specialised search strategies in parallel."""
from __future__ import annotations

import json
import re
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
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": question}],
    )
    raw = resp.content[0].text.strip()

    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE).strip()

    # Attempt JSON parse; fall back to simple keyword queries on failure
    try:
        strategies: Dict[str, List[str]] = json.loads(raw)
    except json.JSONDecodeError:
        # Extract whatever partial arrays are present, then fill gaps
        strategies = {}
        for key in ("systematic_review", "rct", "observational"):
            match = re.search(
                rf'"{key}"\s*:\s*(\[.*?\])', raw, re.DOTALL
            )
            if match:
                try:
                    strategies[key] = json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass

    # Guarantee the three expected keys exist
    for key in ("systematic_review", "rct", "observational"):
        if not strategies.get(key):
            strategies[key] = [question]
    return strategies
