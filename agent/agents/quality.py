"""Quality Agent — scores papers by evidence-hierarchy without an API call."""
from __future__ import annotations

import re
from typing import List

from agent.state import Paper, StudyQuality

# Evidence hierarchy (highest first so the first match wins)
_LEVELS = [
    (5, "Meta-analysis / Systematic Review",
     r"meta.analy|systematic.review|cochrane"),
    (4, "Randomized Controlled Trial",
     r"randomi[sz]ed|randomis|RCT|\bclinical.trial\b|\bplacebo.controlled\b"),
    (3, "Cohort / Prospective Study",
     r"\bcohort\b|prospective|longitudinal|follow.up.stud"),
    (2, "Case-Control / Retrospective",
     r"case.control|retrospective|cross.sectional|population.based"),
    (1, "Case Report / Opinion",
     r"case.report|case.series|editorial|letter.to|commentary|opinion"),
]

_CONFIDENCE = {5: "high", 4: "high", 3: "moderate", 2: "moderate", 1: "low"}


def score_paper(paper: Paper) -> StudyQuality:
    """Classify one paper's study design and return its quality score."""
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
    for level, design, pattern in _LEVELS:
        if re.search(pattern, text, re.IGNORECASE):
            return StudyQuality(
                pmid=paper["pmid"],
                design=design,
                level=level,
                confidence=_CONFIDENCE[level],
            )
    return StudyQuality(
        pmid=paper["pmid"],
        design="Observational / Unclassified",
        level=2,
        confidence="low",
    )


def run(papers: List[Paper]) -> List[StudyQuality]:
    """Score all papers. No API call — pure heuristic classification."""
    return [score_paper(p) for p in papers]


def summary(scores: List[StudyQuality]) -> str:
    """Return a human-readable breakdown of quality levels."""
    counts: dict[int, int] = {}
    for s in scores:
        counts[s["level"]] = counts.get(s["level"], 0) + 1
    parts = []
    labels = {5: "meta-analyses", 4: "RCTs", 3: "cohort", 2: "case-control", 1: "opinion"}
    for lvl in sorted(counts, reverse=True):
        parts.append(f"{counts[lvl]} {labels.get(lvl, 'other')}")
    return " · ".join(parts) if parts else "no papers"
