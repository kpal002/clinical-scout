"""Shared parallel-runner for Quality, Analysis, and Guidelines agents."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

import anthropic

from agent.agents import analysis, guidelines, quality
from agent.state import Contradiction, Finding, GuidelineConflict, Paper, StudyQuality


def run(
    papers: List[Paper],
    question: str,
    mode: str,
    client: anthropic.Anthropic,
) -> Tuple[List[StudyQuality], List[Finding], List[Contradiction], List[GuidelineConflict]]:
    """Run Quality, Analysis, and Guidelines agents concurrently.

    Returns (quality_scores, findings, contradictions, guideline_conflicts).
    """
    quality_scores: List[StudyQuality] = []
    findings: List[Finding] = []
    contradictions: List[Contradiction] = []
    guideline_conflicts: List[GuidelineConflict] = []

    with ThreadPoolExecutor(max_workers=3) as pool:
        q_fut = pool.submit(quality.run, papers)
        a_fut = pool.submit(analysis.run, papers, question, mode, client)
        g_fut = pool.submit(guidelines.run, papers, question, [], client)

        for fut in as_completed([q_fut, a_fut, g_fut]):
            result = fut.result()
            if fut is q_fut:
                quality_scores = result
            elif fut is a_fut:
                findings, contradictions = result
            else:
                guideline_conflicts = result

    return quality_scores, findings, contradictions, guideline_conflicts
