"""Smoke-test the pipeline without a live Anthropic API key."""
import os
import sys
from unittest.mock import MagicMock, patch

from rich.console import Console

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

console = Console()


def _mock_claude_queries(**kwargs):
    """Return pre-baked PubMed queries, context-aware by question content."""
    messages = kwargs.get("messages", [])
    question = messages[0]["content"].lower() if messages else ""
    mock = MagicMock()
    mock.content = [MagicMock()]
    if "knuckle" in question or "arthritis" in question:
        mock.content[0].text = (
            '["knuckle cracking arthritis joint damage",'
            ' "habitual knuckle cracking metacarpophalangeal joint",'
            ' "knuckle cracking osteoarthritis risk meta-analysis"]'
        )
    else:
        mock.content[0].text = (
            '["GLP-1 receptor agonists cardiovascular outcomes meta-analysis",'
            ' "semaglutide liraglutide MACE reduction RCT",'
            ' "GLP-1 agonist heart disease mortality systematic review"]'
        )
    return mock


def _mock_claude_synthesis(**kwargs):
    """Return a stub synthesis, scout or debunker based on the system prompt."""
    system = kwargs.get("system", "")
    mock = MagicMock()
    mock.content = [MagicMock()]
    if "myth debunker" in system.lower():
        mock.content[0].text = (
            "## VERDICT: BUSTED\n\n"
            "## WHAT THE EVIDENCE SAYS\n"
            "Studies show no link between knuckle cracking and arthritis. [MOCK]\n\n"
            "## KEY STUDIES\n- No joint damage observed in long-term studies [1]\n\n"
            "## COMMON MISCONCEPTION\nThe cracking sound is misinterpreted as damage.\n\n"
            "## CITATIONS\n[1] See citation list."
        )
    else:
        mock.content[0].text = (
            "## ANSWER\n"
            "GLP-1 receptor agonists significantly reduce cardiovascular risk. [MOCK]\n\n"
            "## STRENGTH OF EVIDENCE\nStrong — multiple large RCTs.\n\n"
            "## KEY FINDINGS\n- Semaglutide reduced MACE by 26% [1]\n\n"
            "## LIMITATIONS\n- Heterogeneity across trials.\n\n"
            "## CITATIONS\n[1] See citation list."
        )
    return mock


def _run_test(question: str, mode: str) -> None:
    """Run one full pipeline test and assert basic invariants."""
    from agent.tools import reset_client_cache  # pylint: disable=import-outside-toplevel
    from agent.graph import run_agent  # pylint: disable=import-outside-toplevel
    reset_client_cache()  # evict cached client so the mock is picked up fresh

    console.print(f"\n[bold]Mode:[/bold] {mode} | [bold]Question:[/bold] {question}")
    state = run_agent(question, mode=mode)

    console.print(f"  [green]✓[/green] Queries: {state['search_queries']}")
    console.print(f"  [green]✓[/green] Raw papers: {len(state['raw_papers'])}")
    console.print(f"  [green]✓[/green] After σ-RAG: {len(state['filtered_papers'])}")
    console.print(f"  [green]✓[/green] Synthesis: {bool(state['synthesis'])}")
    console.print(f"  [green]✓[/green] Citations: {len(state['citations'])}")

    if mode == "debunker":
        console.print(f"  [green]✓[/green] Verdict: {state.get('verdict')}")
        if state.get("filtered_papers"):
            assert state.get("verdict") in (
                "BUSTED", "SUPPORTED", "MIXED"
            ), "verdict not extracted"


def main() -> None:
    """Run smoke tests for both scout and debunker modes."""
    console.print(
        "\n[bold cyan]Clinical Literature Scout + Medical Myth Debunker"
        " — Smoke Test[/bold cyan]\n"
    )

    os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-fake-key")

    with patch("anthropic.Anthropic") as mock_client:
        instance = mock_client.return_value
        instance.messages.create.side_effect = lambda *a, **kw: (
            _mock_claude_queries(**kw)
            if kw.get("max_tokens", 0) < 1000
            else _mock_claude_synthesis(**kw)
        )

        _run_test(
            "What is the evidence for GLP-1 receptor agonists in reducing cardiovascular risk?",
            mode="scout",
        )
        _run_test(
            "Does cracking your knuckles cause arthritis?",
            mode="debunker",
        )

    console.print("\n[bold green]SMOKE TEST PASSED — both modes verified[/bold green]")


if __name__ == "__main__":
    main()
