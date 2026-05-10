"""Analysis Agent — extracts key findings and detects contradictions via Claude."""
from __future__ import annotations

import json
from typing import List, Tuple

import anthropic

from agent.state import Contradiction, Finding, Paper

_SYSTEM = """You are a biomedical research analyst. Analyze the numbered abstracts below.

Tasks:
1. Extract 3–6 distinct KEY FINDINGS as concise factual claims.
   For each finding note which paper(s) support it and whether the effect is
   positive (supports the treatment/claim), negative (contradicts it), or neutral.

2. Detect CONTRADICTIONS — pairs of papers that reach opposing conclusions on
   the SAME specific topic. Only flag genuine contradictions, not mere nuance.

Return ONLY valid JSON, no other text:
{
  "findings": [
    {"claim": "...", "pmids": ["12345678"], "direction": "positive|negative|neutral"}
  ],
  "contradictions": [
    {
      "topic": "...",
      "finding_a": "...", "pmid_a": "12345678",
      "finding_b": "...", "pmid_b": "87654321"
    }
  ]
}"""

_DEBUNKER_SYSTEM = """You are a medical myth analyst. Analyze the numbered abstracts below.

Tasks:
1. Extract 3–6 KEY FINDINGS relevant to the myth/claim — concise factual claims.
   Mark direction as: positive (evidence supports the myth), negative (busts it),
   or neutral.

2. Detect CONTRADICTIONS — papers that directly conflict on the same specific point.

Return ONLY valid JSON, no other text:
{
  "findings": [
    {"claim": "...", "pmids": ["12345678"], "direction": "positive|negative|neutral"}
  ],
  "contradictions": [
    {
      "topic": "...",
      "finding_a": "...", "pmid_a": "12345678",
      "finding_b": "...", "pmid_b": "87654321"
    }
  ]
}"""


def _build_corpus(papers: List[Paper]) -> str:
    return "\n\n".join(
        f"[{i + 1}] PMID:{p['pmid']} | {p['authors']} ({p['year']}) | {p['journal']}\n"
        f"Title: {p['title']}\n"
        f"Abstract: {p['abstract'][:900]}"
        for i, p in enumerate(papers)
    )


def run(
    papers: List[Paper],
    question: str,
    mode: str,
    client: anthropic.Anthropic,
) -> Tuple[List[Finding], List[Contradiction]]:
    """Extract findings and detect contradictions.

    Returns (findings, contradictions).
    """
    if not papers:
        return [], []

    system = _DEBUNKER_SYSTEM if mode == "debunker" else _SYSTEM
    label = "Medical Myth" if mode == "debunker" else "Clinical Question"
    corpus = _build_corpus(papers)

    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=system,
        messages=[{
            "role": "user",
            "content": f"{label}: {question}\n\nAbstracts:\n{corpus}",
        }],
    )

    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return [], []

    findings: List[Finding] = [
        Finding(
            claim=f.get("claim", ""),
            pmids=f.get("pmids", []),
            direction=f.get("direction", "neutral"),
        )
        for f in data.get("findings", [])
        if f.get("claim")
    ]
    contradictions: List[Contradiction] = [
        Contradiction(
            topic=c.get("topic", ""),
            finding_a=c.get("finding_a", ""),
            pmid_a=c.get("pmid_a", ""),
            finding_b=c.get("finding_b", ""),
            pmid_b=c.get("pmid_b", ""),
        )
        for c in data.get("contradictions", [])
        if c.get("finding_a") and c.get("finding_b")
    ]
    return findings, contradictions
