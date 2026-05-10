"""LangGraph agent graph for the Clinical Literature Scout + Medical Myth Debunker."""
from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph
from rich.console import Console

from agent.state import AgentState
from agent.tools import (
    fetch_abstracts_tool,
    filter_with_sigma_rag,
    formulate_queries,
    search_pubmed_tool,
    synthesize_evidence,
)

console = Console()

MAX_RETRIES = 1


def _route_after_fetch(state: AgentState) -> Literal["sigma_filter", "error_handler"]:
    """Route to error_handler if no papers were retrieved."""
    if not state.get("raw_papers"):
        return "error_handler"
    return "sigma_filter"


def _route_after_sigma(
    state: AgentState,
) -> Literal["synthesizer", "retry_sigma", "error_handler"]:
    """Route to synthesizer, retry, or error based on filtered paper count."""
    if state.get("filtered_papers"):
        return "synthesizer"
    if state.get("retry_count", 0) < MAX_RETRIES:
        return "retry_sigma"
    return "error_handler"


def error_handler(state: AgentState) -> AgentState:
    """Return a graceful error message when no relevant papers are found."""
    msg = (
        "No sufficiently relevant papers were found for this query. "
        "Try rephrasing the clinical question or lowering the σ threshold."
    )
    console.print(f"\n[bold red]\\[ERROR][/bold red] {msg}")
    return {
        **state,
        "synthesis": msg,
        "verdict": None,
        "citations": [],
        "error": msg,
    }


def retry_sigma(state: AgentState) -> AgentState:
    """Lower the σ threshold by 0.5 and re-run the significance filter."""
    retry = state.get("retry_count", 0) + 1
    console.print(
        f"\n[bold yellow]\\[RETRY][/bold yellow] No papers passed σ filter. "
        f"Retrying with lower threshold (retry {retry}/{MAX_RETRIES})..."
    )
    new_state = {**state, "retry_count": retry}
    return filter_with_sigma_rag(new_state, sigma_threshold=1.5)


def build_graph() -> StateGraph:
    """Compile and return the full LangGraph pipeline."""
    builder = StateGraph(AgentState)

    builder.add_node("query_formulator", formulate_queries)
    builder.add_node("pubmed_searcher", search_pubmed_tool)
    builder.add_node("abstract_fetcher", fetch_abstracts_tool)
    builder.add_node(
        "sigma_filter", lambda s: filter_with_sigma_rag(s, sigma_threshold=1.2)
    )
    builder.add_node("retry_sigma", retry_sigma)
    builder.add_node("synthesizer", synthesize_evidence)
    builder.add_node("error_handler", error_handler)

    builder.add_edge(START, "query_formulator")
    builder.add_edge("query_formulator", "pubmed_searcher")
    builder.add_edge("pubmed_searcher", "abstract_fetcher")

    builder.add_conditional_edges(
        "abstract_fetcher",
        _route_after_fetch,
        {"sigma_filter": "sigma_filter", "error_handler": "error_handler"},
    )
    builder.add_conditional_edges(
        "sigma_filter",
        _route_after_sigma,
        {
            "synthesizer": "synthesizer",
            "retry_sigma": "retry_sigma",
            "error_handler": "error_handler",
        },
    )
    builder.add_conditional_edges(
        "retry_sigma",
        _route_after_sigma,
        {
            "synthesizer": "synthesizer",
            "retry_sigma": "error_handler",
            "error_handler": "error_handler",
        },
    )

    builder.add_edge("synthesizer", END)
    builder.add_edge("error_handler", END)

    return builder.compile()


def run_agent(clinical_question: str, mode: str = "scout") -> AgentState:
    """Run the full pipeline and return the final state."""
    graph = build_graph()

    initial_state: AgentState = {
        "clinical_question": clinical_question,
        "messages": [],
        "search_queries": [],
        "raw_papers": [],
        "filtered_papers": [],
        "synthesis": None,
        "verdict": None,
        "citations": [],
        "reasoning_trace": [],
        "mode": mode,
        "step_count": 0,
        "error": None,
        "retry_count": 0,
    }

    return graph.invoke(initial_state)
