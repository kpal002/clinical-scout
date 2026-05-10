"""Smoke-test the multi-agent pipeline without a live Anthropic API key."""
from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

from rich.console import Console

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

console = Console()

# ── Mock Claude responses ─────────────────────────────────────────────────────

_STRATEGIES_GLP1 = json.dumps({
    "systematic_review": [
        "GLP-1 receptor agonists cardiovascular outcomes meta-analysis",
        "semaglutide liraglutide MACE reduction systematic review",
    ],
    "rct": [
        "semaglutide liraglutide randomized controlled trial cardiovascular",
        "GLP-1 agonist heart disease mortality RCT",
    ],
    "observational": [
        "GLP-1 agonist cardiovascular risk cohort study",
        "incretin therapy atherosclerosis observational",
    ],
})

_STRATEGIES_KNUCKLE = json.dumps({
    "systematic_review": [
        "knuckle cracking arthritis systematic review",
        "habitual joint cracking osteoarthritis meta-analysis",
    ],
    "rct": [
        "knuckle cracking joint damage experimental study",
        "metacarpophalangeal cracking clinical trial",
    ],
    "observational": [
        "knuckle cracking long term effects cohort",
        "hand joint cracking radiographic changes observational",
    ],
})

_ANALYSIS_GLP1 = json.dumps({
    "findings": [
        {"claim": "Semaglutide reduced MACE by 26% vs placebo in SUSTAIN-6.",
         "pmids": ["28095653"], "direction": "positive"},
        {"claim": "Liraglutide reduced CV death by 22% in LEADER trial.",
         "pmids": ["26378978"], "direction": "positive"},
    ],
    "contradictions": [],
})

_ANALYSIS_KNUCKLE = json.dumps({
    "findings": [
        {"claim": "No association found between knuckle cracking and arthritis in case-control.",
         "pmids": ["11500798"], "direction": "negative"},
        {"claim": "Habitual cracking did not cause detectable joint damage on X-ray.",
         "pmids": ["18528959"], "direction": "negative"},
    ],
    "contradictions": [],
})

_SYNTHESIS_SCOUT = (
    "## ANSWER\n"
    "GLP-1 receptor agonists significantly reduce cardiovascular risk. [MOCK]\n\n"
    "## STRENGTH OF EVIDENCE\nStrong — multiple large RCTs.\n\n"
    "## KEY FINDINGS\n- Semaglutide reduced MACE by 26% [1]\n\n"
    "## CONTRADICTIONS\nNone detected.\n\n"
    "## LIMITATIONS\n- Heterogeneity across trials.\n\n"
    "## CITATIONS\n[1] See citation list."
)

_SYNTHESIS_DEBUNKER = (
    "## VERDICT: BUSTED\n\n"
    "## WHAT THE EVIDENCE SAYS\n"
    "Studies show no link between knuckle cracking and arthritis. [MOCK]\n\n"
    "## KEY STUDIES\n- No joint damage observed [1]\n\n"
    "## CONTRADICTIONS\nNone detected.\n\n"
    "## COMMON MISCONCEPTION\nThe sound is misinterpreted as damage.\n\n"
    "## CITATIONS\n[1] See citation list."
)


def _mock_create(**kwargs) -> MagicMock:
    """Dispatch mock Claude response based on call signature."""
    system = kwargs.get("system", "")
    messages = kwargs.get("messages", [])
    user_content = messages[0]["content"].lower() if messages else ""

    mock = MagicMock()
    mock.content = [MagicMock()]

    # Query Agent call → JSON strategies
    if "search strateg" in system.lower() or "mesh query" in system.lower():
        if "knuckle" in user_content or "arthritis" in user_content:
            mock.content[0].text = _STRATEGIES_KNUCKLE
        else:
            mock.content[0].text = _STRATEGIES_GLP1

    # Analysis Agent call → JSON findings + contradictions
    elif '"findings"' in system or "findings" in system.lower() and "contradictions" in system.lower():
        if "knuckle" in user_content or "arthritis" in user_content:
            mock.content[0].text = _ANALYSIS_KNUCKLE
        else:
            mock.content[0].text = _ANALYSIS_GLP1

    # Synthesis Agent call → structured text
    elif "myth debunker" in system.lower() or "debunker" in system.lower():
        mock.content[0].text = _SYNTHESIS_DEBUNKER
    else:
        mock.content[0].text = _SYNTHESIS_SCOUT

    return mock


# ── Test runner ───────────────────────────────────────────────────────────────

def _run_test(question: str, mode: str) -> None:
    from agent.tools import reset_client_cache  # pylint: disable=import-outside-toplevel
    from agent.graph import run_agent            # pylint: disable=import-outside-toplevel
    reset_client_cache()

    console.print(f"\n[bold]Mode:[/bold] {mode} | [bold]Question:[/bold] {question}")
    state = run_agent(question, mode=mode)

    console.print(f"  [green]✓[/green] Strategies: {list(state.get('search_strategies', {}).keys())}")
    console.print(f"  [green]✓[/green] Queries: {len(state.get('search_queries', []))}")
    console.print(f"  [green]✓[/green] Raw papers: {len(state['raw_papers'])}")
    console.print(f"  [green]✓[/green] After σ-RAG: {len(state['filtered_papers'])}")
    console.print(f"  [green]✓[/green] Quality scores: {len(state.get('quality_scores', []))}")
    console.print(f"  [green]✓[/green] Findings: {len(state.get('findings', []))}")
    console.print(f"  [green]✓[/green] Contradictions: {len(state.get('contradictions', []))}")
    console.print(f"  [green]✓[/green] Synthesis: {bool(state['synthesis'])}")
    console.print(f"  [green]✓[/green] Citations: {len(state['citations'])}")

    if mode == "debunker" and state.get("filtered_papers"):
        verdict = state.get("verdict")
        console.print(f"  [green]✓[/green] Verdict: {verdict}")
        assert verdict in ("BUSTED", "SUPPORTED", "MIXED"), f"unexpected verdict: {verdict}"


def main() -> None:
    console.print(
        "\n[bold cyan]Clinical Literature Scout — Multi-Agent Smoke Test[/bold cyan]\n"
    )
    os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-fake-key")

    with patch("anthropic.Anthropic") as mock_cls:
        instance = mock_cls.return_value
        instance.messages.create.side_effect = lambda *a, **kw: _mock_create(**kw)

        _run_test(
            "What is the evidence for GLP-1 receptor agonists in reducing cardiovascular risk?",
            mode="scout",
        )
        _run_test(
            "Does cracking your knuckles cause arthritis?",
            mode="debunker",
        )

    console.print("\n[bold green]SMOKE TEST PASSED — multi-agent pipeline verified[/bold green]")


if __name__ == "__main__":
    main()
