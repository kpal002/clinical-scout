#!/usr/bin/env python3
"""Clinical Literature Scout + Medical Myth Debunker — entry point.

Supports two modes: Scout (clinical questions) and Debunker (myth verification),
toggled via a single --mode flag.

Usage::

    python main.py --mode scout --question "Does metformin reduce cancer risk?"
    python main.py --mode debunker --question "Does cracking knuckles cause arthritis?"
    python main.py --interactive
"""
from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Path bootstrap — must precede first-party imports so the package is found
# when the script is run directly from the clinical-scout directory.
# pylint: disable=wrong-import-position
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _path in (_HERE, _REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dotenv import load_dotenv  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.prompt import Prompt  # noqa: E402

from agent.graph import run_agent  # noqa: E402
from ui.terminal import print_header, print_synthesis  # noqa: E402

load_dotenv()

console = Console()


def _require_api_key() -> None:
    """Exit with a helpful message when ANTHROPIC_API_KEY is missing."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[bold red]Error:[/bold red] ANTHROPIC_API_KEY is not set.")
        console.print("Set it with: export ANTHROPIC_API_KEY=your_key")
        sys.exit(1)


def run_cli(question: str, mode: str) -> None:
    """Run a single question in non-interactive CLI mode."""
    _require_api_key()
    print_header(question, mode)
    state = run_agent(question, mode=mode)
    print_synthesis(state)


def run_interactive() -> None:
    """Prompt the user for mode and question in a REPL loop."""
    _require_api_key()
    console.print(
        "\n[bold cyan]Clinical Literature Scout + Medical Myth Debunker[/bold cyan]"
        " — Interactive Mode"
    )
    console.print("[dim]Type 'quit' to exit[/dim]\n")
    while True:
        mode = Prompt.ask(
            "[bold]Mode[/bold]",
            choices=["scout", "debunker"],
            default="scout",
        )
        prompt_text = (
            "[bold]Enter clinical question[/bold]"
            if mode == "scout"
            else "[bold]Enter medical myth to debunk[/bold]"
        )
        question = Prompt.ask(prompt_text).strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        print_header(question, mode)
        state = run_agent(question, mode=mode)
        print_synthesis(state)
        console.print("\n" + "─" * 60 + "\n")


def run_web(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the optional FastAPI web server."""
    _require_api_key()
    try:
        import uvicorn  # pylint: disable=import-outside-toplevel
        from ui.web import app  # pylint: disable=import-outside-toplevel
    except ImportError:
        console.print(
            "[red]FastAPI/Uvicorn not installed. Run: pip install fastapi uvicorn[/red]"
        )
        sys.exit(1)
    console.print(f"Starting web server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate mode."""
    parser = argparse.ArgumentParser(
        description=(
            "Clinical Literature Scout + Medical Myth Debunker\n"
            "Supports two modes: Scout (clinical questions) and Debunker "
            "(myth verification), toggled via the --mode flag."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["scout", "debunker"],
        default="scout",
        help=(
            "scout: evidence-based clinical answers | "
            "debunker: myth verification (default: scout)"
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--question", "-q", type=str, help="Question or myth (CLI mode)")
    group.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    group.add_argument("--web", "-w", action="store_true", help="Start FastAPI web server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.web:
        run_web(args.host, args.port)
    elif args.interactive:
        run_interactive()
    elif args.question:
        run_cli(args.question, args.mode)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
