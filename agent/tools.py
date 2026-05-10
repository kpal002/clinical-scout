"""LangGraph tool definitions for the Clinical Literature Scout + Medical Myth Debunker."""
from __future__ import annotations

import functools
import json
import os
from typing import List

import anthropic
from rich.console import Console

from agent.state import AgentState, Paper
from retrieval import pubmed, sigma_filter

console = Console()


@functools.lru_cache(maxsize=1)
def _client() -> anthropic.Anthropic:
    """Return a cached Anthropic client (one instance per process)."""
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def reset_client_cache() -> None:
    """Evict the cached Anthropic client (useful in tests to inject mocks)."""
    _client.cache_clear()


# ---------------------------------------------------------------------------
# Tool 1: formulate_queries
# ---------------------------------------------------------------------------

def formulate_queries(state: AgentState) -> AgentState:
    """Translate clinical question into optimised PubMed search queries."""
    question = state["clinical_question"]
    console.print(
        "\n[bold yellow]\\[STEP 1][/bold yellow] Formulating PubMed search queries..."
    )

    response = _client().messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        system=(
            "You are a biomedical librarian expert in PubMed MeSH query formulation. "
            "Given a clinical question, produce exactly 3 optimised PubMed search queries "
            "that together cover the topic broadly. Prioritise meta-analyses, systematic "
            "reviews, and RCTs. Return ONLY a JSON array of 3 strings, e.g. "
            '["query1", "query2", "query3"]. No other text.'
        ),
        messages=[{"role": "user", "content": question}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    queries: List[str] = json.loads(raw)

    for i, q in enumerate(queries, 1):
        console.print(f"  [green]→[/green] Query {i}: [italic]{q}[/italic]")

    trace = list(state.get("reasoning_trace", []))
    trace.append(f"STEP 1: Formulated {len(queries)} PubMed queries for: {question}")

    return {
        **state,
        "search_queries": queries,
        "step_count": state.get("step_count", 0) + 1,
        "reasoning_trace": trace,
    }


# ---------------------------------------------------------------------------
# Tool 2: search_pubmed
# ---------------------------------------------------------------------------

def search_pubmed_tool(state: AgentState) -> AgentState:
    """Search PubMed for each query and collect unique PMIDs."""
    queries = state["search_queries"]
    console.print("\n[bold yellow]\\[STEP 2][/bold yellow] Searching PubMed...")

    all_pmids: set[str] = set()
    for q in queries:
        pmids = pubmed.search_pubmed(q, max_results=20)
        console.print(
            f"  [green]→[/green] \"{q[:60]}\": [cyan]{len(pmids)}[/cyan] results"
        )
        all_pmids.update(pmids)

    unique_pmids = list(all_pmids)
    console.print(
        f"  [bold]→ Total unique PMIDs: [cyan]{len(unique_pmids)}[/cyan][/bold]"
    )

    trace = list(state.get("reasoning_trace", []))
    trace.append(
        f"STEP 2: Retrieved {len(unique_pmids)} unique PMIDs across {len(queries)} queries"
    )

    placeholder = Paper(
        pmid="", title="", abstract="", authors="",
        year="", journal="", url="", sigma_score=None,
    )
    return {
        **state,
        "raw_papers": [
            Paper(**{**placeholder, "pmid": p}) for p in unique_pmids
        ],
        "step_count": state.get("step_count", 0) + 1,
        "reasoning_trace": trace,
        "_pmids": unique_pmids,
    }


# ---------------------------------------------------------------------------
# Tool 3: fetch_abstracts
# ---------------------------------------------------------------------------

def fetch_abstracts_tool(state: AgentState) -> AgentState:
    """Fetch full metadata and abstracts for collected PMIDs."""
    pmids = state.get("_pmids") or [p["pmid"] for p in state.get("raw_papers", [])]
    console.print(
        f"\n[bold yellow]\\[STEP 3][/bold yellow] "
        f"Fetching abstracts for {len(pmids)} papers..."
    )

    papers = pubmed.fetch_abstracts(pmids)
    no_abstract = len(pmids) - len(papers)

    console.print(
        f"  [green]→[/green] Fetched [cyan]{len(papers)}[/cyan] abstracts "
        f"([dim]{no_abstract} papers had no abstract[/dim])"
    )

    trace = list(state.get("reasoning_trace", []))
    trace.append(f"STEP 3: Fetched {len(papers)} abstracts ({no_abstract} had none)")

    return {
        **state,
        "raw_papers": papers,
        "step_count": state.get("step_count", 0) + 1,
        "reasoning_trace": trace,
    }


# ---------------------------------------------------------------------------
# Tool 4: filter_with_sigma_rag
# ---------------------------------------------------------------------------

def filter_with_sigma_rag(state: AgentState, sigma_threshold: float = 1.2) -> AgentState:
    """Apply σ-RAG significance filtering to the paper corpus."""
    papers = state["raw_papers"]
    question = state["clinical_question"]
    retry = state.get("retry_count", 0)

    if retry > 0:
        sigma_threshold = max(1.0, sigma_threshold - 0.5 * retry)

    console.print(
        f"\n[bold yellow]\\[STEP 4][/bold yellow] "
        f"Applying σ-RAG significance filter "
        f"(threshold: [cyan]{sigma_threshold}σ[/cyan])..."
    )
    console.print(
        f"  [dim]→ Fitting Gaussian noise floor on {len(papers)}-paper corpus...[/dim]"
    )

    def _progress(title: str, score: float, passed: bool) -> None:
        short = title[:60] + "..." if len(title) > 60 else title
        if passed:
            console.print(
                f"  [green]✓[/green] [dim]{short}[/dim] "
                f"→ σ = [bold green]{score:.2f}[/bold green]"
            )
        else:
            console.print(
                f"  [red]✗[/red] [dim]{short}[/dim] "
                f"→ σ = [dim red]{score:.2f}[/dim red]"
            )

    passed_papers = sigma_filter.filter_papers(
        papers,
        question,
        sigma_threshold=sigma_threshold,
        max_results=10,
        progress_callback=_progress,
    )
    filtered_count = len(papers) - len(passed_papers)

    console.print(
        f"\n  [bold]→ {len(papers)} papers → "
        f"[green]{len(passed_papers)}[/green] passed σ threshold → "
        f"[red]{filtered_count}[/red] filtered out[/bold]"
    )

    trace = list(state.get("reasoning_trace", []))
    trace.append(
        f"STEP 4: σ-RAG filtered {len(papers)} → "
        f"{len(passed_papers)} papers at {sigma_threshold}σ"
    )

    return {
        **state,
        "filtered_papers": passed_papers,
        "step_count": state.get("step_count", 0) + 1,
        "reasoning_trace": trace,
    }


# ---------------------------------------------------------------------------
# Tool 5: synthesize_evidence
# ---------------------------------------------------------------------------

_SCOUT_SYSTEM_PROMPT = (
    "You are a clinical evidence synthesizer. Given a clinical question and a set of "
    "filtered, relevant abstracts, produce a structured synthesis using EXACTLY this format:\n\n"
    "## ANSWER\n"
    "(Direct answer to the clinical question, 2–3 sentences)\n\n"
    "## STRENGTH OF EVIDENCE\n"
    "(Strong / Moderate / Weak / Insufficient) — one sentence justification\n\n"
    "## KEY FINDINGS\n"
    "- Bullet points of the most important findings, each citing [N]\n\n"
    "## LIMITATIONS\n"
    "- Key gaps or contradictions in the evidence\n\n"
    "## CITATIONS\n"
    "[1] Authors (Year). Title. Journal. URL\n"
    "(list all papers used)\n\n"
    "Rules:\n"
    "- Cite specific papers for specific claims using [N]\n"
    "- Flag conflicting evidence explicitly\n"
    "- Never make claims not supported by the provided abstracts\n"
    "- Be precise and concise"
)

_DEBUNKER_SYSTEM_PROMPT = (
    "You are a medical myth debunker. Given a myth and filtered abstracts, produce a structured "
    "response using EXACTLY this format:\n\n"
    "## VERDICT: BUSTED / SUPPORTED / MIXED EVIDENCE\n"
    "(one word on the same line as VERDICT, in bold)\n\n"
    "## WHAT THE EVIDENCE SAYS\n"
    "(2–3 sentences summarizing what the science actually shows)\n\n"
    "## KEY STUDIES\n"
    "- Bullet points of the most relevant findings, each citing [N]\n\n"
    "## COMMON MISCONCEPTION\n"
    "(Why people believe the myth — 1–2 sentences)\n\n"
    "## CITATIONS\n"
    "[1] Authors (Year). Title. Journal. URL\n"
    "(list all papers used)\n\n"
    "Rules:\n"
    "- Be direct. Lead with the verdict.\n"
    "- Write for a general audience — avoid jargon\n"
    "- Cite specific papers for specific claims using [N]\n"
    "- Never make claims not supported by the provided abstracts"
)


def _extract_verdict(synthesis: str) -> str | None:
    """Scan the first VERDICT line and return BUSTED, SUPPORTED, or MIXED."""
    for line in synthesis.splitlines():
        upper = line.upper()
        if "BUSTED" in upper:
            return "BUSTED"
        if "SUPPORTED" in upper:
            return "SUPPORTED"
        if "MIXED" in upper:
            return "MIXED"
    return None


def synthesize_evidence(state: AgentState) -> AgentState:
    """Generate structured evidence synthesis with citations."""
    papers = state["filtered_papers"]
    question = state["clinical_question"]
    mode = state.get("mode", "scout")

    console.print(
        f"\n[bold yellow]\\[STEP 5][/bold yellow] "
        f"Synthesizing evidence from [cyan]{len(papers)}[/cyan] papers..."
    )
    console.print(
        "  [dim]→ Calling Claude claude-sonnet-4-5 for evidence synthesis...[/dim]"
    )

    numbered_abstracts = "\n\n".join(
        f"[{i + 1}] PMID:{p['pmid']} | {p['authors']} ({p['year']}) | {p['journal']}\n"
        f"Title: {p['title']}\n"
        f"Abstract: {p['abstract'][:1500]}"
        for i, p in enumerate(papers)
    )

    system_prompt = _DEBUNKER_SYSTEM_PROMPT if mode == "debunker" else _SCOUT_SYSTEM_PROMPT
    user_label = "Medical Myth" if mode == "debunker" else "Clinical Question"

    response = _client().messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2048,
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": f"{user_label}: {question}\n\nRelevant Papers:\n{numbered_abstracts}",
        }],
    )

    synthesis = response.content[0].text.strip()
    verdict = _extract_verdict(synthesis) if mode == "debunker" else None

    citations = [
        f"[{i + 1}] {p['authors']} ({p['year']}). {p['title']}. {p['journal']}. {p['url']}"
        for i, p in enumerate(papers)
    ]

    trace = list(state.get("reasoning_trace", []))
    trace.append(f"STEP 5: Synthesized evidence from {len(papers)} papers (mode={mode})")

    return {
        **state,
        "synthesis": synthesis,
        "verdict": verdict,
        "citations": citations,
        "step_count": state.get("step_count", 0) + 1,
        "reasoning_trace": trace,
    }
