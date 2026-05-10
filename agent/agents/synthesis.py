"""Synthesis Agent — assembles claim-level citations using quality + analysis context."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import anthropic

from agent.state import Contradiction, Finding, Paper, StudyQuality

_SCOUT_SYSTEM = """You are a clinical evidence synthesizer. Produce a structured report.

RULES:
- Every factual claim MUST cite specific papers using inline [N] notation.
- Weight higher-quality evidence (meta-analyses > RCTs > cohort studies).
- Explicitly flag contradictions where they exist.
- Never fabricate — only use information from the provided abstracts.

FORMAT (use exactly these headers):
## ANSWER
(Direct answer, 2–3 sentences with [N] citations)

## STRENGTH OF EVIDENCE
(Strong / Moderate / Weak / Insufficient) — one sentence justifying the rating

## KEY FINDINGS
- Each finding on its own line, citing [N]

## CONTRADICTIONS
- Any conflicting findings between papers, or "None detected."

## LIMITATIONS
- Key gaps, biases, or caveats

## CITATIONS
[1] Authors (Year). Title. Journal."""

_DEBUNKER_SYSTEM = """You are a medical myth debunker. Produce a structured verdict.

RULES:
- Every factual claim MUST cite specific papers using inline [N] notation.
- Lead with a clear verdict on the first line of ## VERDICT.
- Weight higher-quality evidence when reaching the verdict.
- Explicitly flag contradictions where they exist.
- Never fabricate — only use information from the provided abstracts.
- Write for a general audience.

FORMAT (use exactly these headers):
## VERDICT: BUSTED / SUPPORTED / MIXED EVIDENCE
(2–3 sentences explaining the verdict with [N] citations)

## WHAT THE EVIDENCE SAYS
(Concise summary with [N] citations)

## KEY STUDIES
- Bullet points with [N] citations

## CONTRADICTIONS
- Any conflicting findings, or "None detected."

## COMMON MISCONCEPTION
(Why people believe the myth — 1–2 sentences)

## CITATIONS
[1] Authors (Year). Title. Journal."""


def _quality_context(
    papers: List[Paper],
    scores: List[StudyQuality],
) -> str:
    score_map: Dict[str, StudyQuality] = {s["pmid"]: s for s in scores}
    lines = []
    for i, p in enumerate(papers, 1):
        sq = score_map.get(p["pmid"])
        design = sq["design"] if sq else "Unclassified"
        level = sq["level"] if sq else "?"
        lines.append(f"  [{i}] Level {level} — {design}")
    return "\n".join(lines)


def _findings_context(findings: List[Finding]) -> str:
    if not findings:
        return "  None extracted."
    dir_icon = {"positive": "↑", "negative": "↓", "neutral": "→"}
    return "\n".join(
        f"  {dir_icon.get(f['direction'], '→')} {f['claim']} "
        f"(PMIDs: {', '.join(f['pmids'])})"
        for f in findings
    )


def _contradictions_context(contradictions: List[Contradiction]) -> str:
    if not contradictions:
        return "  None detected."
    return "\n".join(
        f"  ⚡ {c['topic']}: PMID {c['pmid_a']} vs PMID {c['pmid_b']}"
        for c in contradictions
    )


def _build_prompt(
    papers: List[Paper],
    scores: List[StudyQuality],
    findings: List[Finding],
    contradictions: List[Contradiction],
    question: str,
    mode: str,
) -> str:
    label = "Medical Myth" if mode == "debunker" else "Clinical Question"
    numbered = "\n\n".join(
        f"[{i + 1}] PMID:{p['pmid']} | {p['authors']} ({p['year']}) | {p['journal']}\n"
        f"Title: {p['title']}\n"
        f"Abstract: {p['abstract'][:1200]}"
        for i, p in enumerate(papers)
    )
    return (
        f"{label}: {question}\n\n"
        f"=== EVIDENCE QUALITY ===\n{_quality_context(papers, scores)}\n\n"
        f"=== KEY FINDINGS PRE-EXTRACTED ===\n{_findings_context(findings)}\n\n"
        f"=== CONTRADICTIONS DETECTED ===\n{_contradictions_context(contradictions)}\n\n"
        f"=== PAPERS (cite by number) ===\n{numbered}"
    )


def _extract_verdict(synthesis: str) -> Optional[str]:
    for line in synthesis.splitlines():
        upper = line.upper()
        if "BUSTED" in upper:
            return "BUSTED"
        if "SUPPORTED" in upper:
            return "SUPPORTED"
        if "MIXED" in upper:
            return "MIXED"
    return None


def run(
    papers: List[Paper],
    scores: List[StudyQuality],
    findings: List[Finding],
    contradictions: List[Contradiction],
    question: str,
    mode: str,
    client: anthropic.Anthropic,
) -> Tuple[str, Optional[str], List[str]]:
    """Produce the final synthesis with claim-level citations.

    Returns (synthesis_text, verdict_or_None, citation_list).
    """
    system = _DEBUNKER_SYSTEM if mode == "debunker" else _SCOUT_SYSTEM
    prompt = _build_prompt(papers, scores, findings, contradictions, question, mode)

    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    synthesis = resp.content[0].text.strip()
    verdict = _extract_verdict(synthesis) if mode == "debunker" else None
    citations = [
        f"[{i + 1}] {p['authors']} ({p['year']}). {p['title']}. {p['journal']}. {p['url']}"
        for i, p in enumerate(papers)
    ]
    return synthesis, verdict, citations
