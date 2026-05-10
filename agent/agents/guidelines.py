"""Guidelines Agent — compare retrieved evidence against major clinical guidelines."""
from __future__ import annotations

import json
import re
from typing import List

import anthropic

from agent.state import Finding, GuidelineConflict, Paper
from retrieval import pubmed

_MAX_GUIDELINE_RESULTS = 5
_ORGS = ["AHA", "ADA", "USPSTF", "NICE", "Cochrane", "ESC", "WHO"]

_SYSTEM = """You are a clinical guideline analyst. Given research findings and retrieved
guideline documents, identify where emerging evidence supports or conflicts with
current recommendations.

Return ONLY valid JSON — an array of objects with these exact keys:
  organization   (string: e.g. "AHA", "ADA", "USPSTF", "NICE", "Cochrane")
  recommendation (string: what the guideline recommends)
  evidence_summary (string: what the retrieved research papers show)
  conflict_level (string: one of "supports", "minor", "moderate", "major")
  pmid           (string: PMID of the guideline document)

Rules:
- Only create entries when you can cite a specific guideline PMID.
- conflict_level="supports" means the new evidence confirms the guideline.
- conflict_level="major" means the new evidence strongly contradicts the guideline.
- Be precise — quote specific numbers or recommendations when available.
- Return [] if no relevant guidelines were found.
"""


def _build_guideline_query(question: str) -> str:
    """Append PubMed guideline publication-type filter to the question keywords."""
    # Strip very common stop words to keep the query focused
    stop = {"the", "a", "an", "of", "for", "in", "is", "are", "does", "do", "and", "or"}
    keywords = " ".join(
        w for w in re.sub(r"[^\w\s]", " ", question).split()
        if w.lower() not in stop
    )
    return f"{keywords} [pt]guideline"


def _fetch_guideline_papers(question: str) -> List[Paper]:
    """Search PubMed for guideline-type documents relevant to the question."""
    query = _build_guideline_query(question)
    try:
        pmids = pubmed.search_pubmed(query, max_results=_MAX_GUIDELINE_RESULTS)
    except Exception:  # pylint: disable=broad-exception-caught
        return []
    if not pmids:
        return []
    try:
        return pubmed.fetch_abstracts(pmids)
    except Exception:  # pylint: disable=broad-exception-caught
        return []


def _build_prompt(
    question: str,
    guideline_papers: List[Paper],
    findings: List[Finding],
) -> str:
    if not guideline_papers:
        return ""

    guidelines_text = "\n\n".join(
        f"[G{i + 1}] PMID:{p['pmid']} | {p['authors']} ({p['year']}) | {p['journal']}\n"
        f"Title: {p['title']}\n"
        f"Abstract: {p['abstract'][:800]}"
        for i, p in enumerate(guideline_papers)
    )

    findings_text = (
        "\n".join(
            f"  - {f['claim']} ({f['direction']}, PMIDs: {', '.join(f['pmids'])})"
            for f in findings
        )
        or "  None available."
    )

    return (
        f"Clinical Question: {question}\n\n"
        f"=== RESEARCH FINDINGS (from retrieved evidence) ===\n{findings_text}\n\n"
        f"=== GUIDELINE DOCUMENTS ===\n{guidelines_text}\n\n"
        "Compare the research findings against the guidelines above. "
        "Return a JSON array of GuidelineConflict objects as specified."
    )


def run(
    papers: List[Paper],
    question: str,
    findings: List[Finding],
    client: anthropic.Anthropic,
) -> List[GuidelineConflict]:
    """Search PubMed guidelines and compare against research findings.

    Returns a list of GuidelineConflict dicts, possibly empty.
    """
    _ = papers  # kept for API symmetry; may use for context in future
    guideline_papers = _fetch_guideline_papers(question)
    if not guideline_papers:
        return []

    prompt = _build_prompt(question, guideline_papers, findings)
    if not prompt:
        return []

    try:
        resp = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
        data = json.loads(raw)
        conflicts: List[GuidelineConflict] = []
        for item in data:
            if isinstance(item, dict) and "organization" in item:
                conflicts.append(
                    GuidelineConflict(
                        organization=str(item.get("organization", "")),
                        recommendation=str(item.get("recommendation", "")),
                        evidence_summary=str(item.get("evidence_summary", "")),
                        conflict_level=str(item.get("conflict_level", "supports")),
                        pmid=str(item.get("pmid", "")),
                    )
                )
        return conflicts
    except Exception:  # pylint: disable=broad-exception-caught
        return []
