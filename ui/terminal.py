"""Rich terminal UI — header, verdict banner, and synthesis panel."""
from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from agent.state import AgentState

console = Console()

_VERDICT_STYLE = {
    "BUSTED": ("🟢 BUSTED", "bold green"),
    "SUPPORTED": ("🔴 SUPPORTED", "bold red"),
    "MIXED": ("🟡 MIXED EVIDENCE", "bold yellow"),
}


def print_header(question: str, mode: str = "scout") -> None:
    """Print the application header panel with mode label and question."""
    mode_label = (
        "Medical Myth Debunker" if mode == "debunker" else "Clinical Literature Scout"
    )
    console.print()
    console.print(
        Panel(
            Text(question, style="bold white"),
            title=f"[bold cyan]{mode_label}[/bold cyan]",
            subtitle="[dim]Powered by σ-RAG + PubMed + Claude[/dim]",
            border_style="cyan",
            padding=(1, 2),
        )
    )


def print_verdict(verdict: str) -> None:
    """Print a colour-coded verdict banner for debunker mode."""
    label, style = _VERDICT_STYLE.get(verdict, (verdict, "bold white"))
    colour = style.split()[-1]
    console.print()
    console.print(
        Panel(
            Text(f"VERDICT:  {label}", style=style, justify="center"),
            border_style=colour,
            padding=(0, 4),
        )
    )


def print_synthesis(state: AgentState) -> None:
    """Print the full synthesis, citations, and run-stats panel."""
    mode = state.get("mode", "scout")
    rule_title = (
        "MYTH DEBUNK REPORT" if mode == "debunker" else "CLINICAL EVIDENCE SYNTHESIS"
    )

    verdict = state.get("verdict")
    if mode == "debunker" and verdict:
        print_verdict(verdict)

    console.print()
    console.print(Rule(f"[bold cyan]{rule_title}[/bold cyan]", style="cyan"))
    console.print()

    synthesis = state.get("synthesis", "No synthesis available.")
    if synthesis:
        console.print(Markdown(synthesis))

    console.print()
    console.print(Rule("[bold cyan]CITATIONS[/bold cyan]", style="cyan"))
    console.print()

    for citation in state.get("citations", []):
        console.print(f"  {citation}")

    console.print()
    filtered_n = len(state.get("filtered_papers", []))
    raw_n = len(state.get("raw_papers", []))
    console.print(
        Panel(
            f"[bold]Papers retrieved:[/bold] {raw_n}  "
            f"[bold]Papers after σ-RAG filter:[/bold] {filtered_n}  "
            f"[bold]Steps:[/bold] {state.get('step_count', 0)}  "
            f"[bold]Mode:[/bold] {mode}",
            style="dim",
            border_style="dim",
        )
    )
