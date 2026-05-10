"""Multi-agent LangGraph pipeline — Clinical Literature Scout + Medical Myth Debunker.

Graph topology:
  START
    → query_agent       (Query Agent: 3 specialised strategies)
    → retrieval_agent   (Retrieval Agent: parallel searches + σ-RAG)
    → parallel_agents   (Quality Agent ∥ Analysis Agent via ThreadPoolExecutor)
    → synthesis_agent   (Synthesis Agent: claim-level citations)
  END
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import anthropic
from langgraph.graph import END, START, StateGraph

from agent.agents import analysis, guidelines, quality, query, retrieval, synthesis
from agent.agents.synthesis import SynthesisInput
from agent.state import AgentState, Contradiction, Finding, GuidelineConflict, StudyQuality


# ── Shared client ─────────────────────────────────────────────────────────────

def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


# ── Node: Query Agent ─────────────────────────────────────────────────────────

def orchestrate_query(state: AgentState) -> AgentState:
    """Query Agent: translate question into 3 specialised search strategies."""
    question = state["clinical_question"]
    mode = state.get("mode", "scout")

    strategies = query.run(question, mode, _client())
    flat_queries: List[str] = [q for qs in strategies.values() for q in qs]

    trace = list(state.get("reasoning_trace", []))
    trace.append(
        f"Query Agent: {len(strategies)} strategies, {len(flat_queries)} queries total"
    )
    return {
        **state,
        "search_strategies": strategies,
        "search_queries": flat_queries,
        "step_count": state.get("step_count", 0) + 1,
        "reasoning_trace": trace,
    }


# ── Node: Retrieval Agent ─────────────────────────────────────────────────────

def orchestrate_retrieval(state: AgentState) -> AgentState:
    """Retrieval Agent: run parallel PubMed searches and apply σ-RAG filter."""
    strategies = state["search_strategies"]
    question = state["clinical_question"]

    raw, filtered = retrieval.run(strategies, question)

    trace = list(state.get("reasoning_trace", []))
    trace.append(
        f"Retrieval Agent: {len(raw)} raw → {len(filtered)} passed σ-RAG"
    )
    return {
        **state,
        "raw_papers": raw,
        "filtered_papers": filtered,
        "step_count": state.get("step_count", 0) + 1,
        "reasoning_trace": trace,
    }


# ── Node: Parallel Quality + Analysis ────────────────────────────────────────

def orchestrate_parallel(state: AgentState) -> AgentState:
    """Run Quality, Analysis, and Guidelines agents concurrently."""
    papers = state["filtered_papers"]
    question = state["clinical_question"]
    mode = state.get("mode", "scout")
    client = _client()

    quality_scores: List[StudyQuality] = []
    findings: List[Finding] = []
    contradictions: List[Contradiction] = []
    guideline_conflicts: List[GuidelineConflict] = []

    with ThreadPoolExecutor(max_workers=3) as pool:
        q_future = pool.submit(quality.run, papers)
        a_future = pool.submit(analysis.run, papers, question, mode, client)
        # Guidelines run after analysis resolves findings; use empty list for now
        # (guidelines.run accepts findings for context but can run with [])
        g_future = pool.submit(guidelines.run, papers, question, [], client)

        for future in as_completed([q_future, a_future, g_future]):
            result = future.result()
            if future is q_future:
                quality_scores = result
            elif future is a_future:
                findings, contradictions = result
            else:
                guideline_conflicts = result

    trace = list(state.get("reasoning_trace", []))
    trace.append(
        f"Quality Agent: {len(quality_scores)} papers scored | "
        f"Analysis Agent: {len(findings)} findings, {len(contradictions)} contradictions | "
        f"Guidelines Agent: {len(guideline_conflicts)} conflicts"
    )
    return {
        **state,
        "quality_scores": quality_scores,
        "findings": findings,
        "contradictions": contradictions,
        "guideline_conflicts": guideline_conflicts,
        "step_count": state.get("step_count", 0) + 1,
        "reasoning_trace": trace,
    }


# ── Node: Synthesis Agent ─────────────────────────────────────────────────────

def orchestrate_synthesis(state: AgentState) -> AgentState:
    """Synthesis Agent: assemble claim-level citations from quality + analysis output."""
    papers = state["filtered_papers"]
    scores = state.get("quality_scores", [])
    all_findings = state.get("findings", [])
    all_contradictions = state.get("contradictions", [])
    question = state["clinical_question"]
    mode = state.get("mode", "scout")

    all_guideline_conflicts = state.get("guideline_conflicts", [])
    synth, verdict, cites = synthesis.run(
        SynthesisInput(
            papers=papers, scores=scores, findings=all_findings,
            contradictions=all_contradictions,
            guideline_conflicts=all_guideline_conflicts,
            question=question, mode=mode,
        ),
        _client(),
    )

    trace = list(state.get("reasoning_trace", []))
    trace.append(f"Synthesis Agent: report assembled from {len(papers)} papers")
    return {
        **state,
        "synthesis": synth,
        "verdict": verdict,
        "citations": cites,
        "step_count": state.get("step_count", 0) + 1,
        "reasoning_trace": trace,
    }


# ── Error handler ─────────────────────────────────────────────────────────────

def error_handler(state: AgentState) -> AgentState:
    """Clear output fields when the pipeline cannot produce results."""
    return {
        **state,
        "synthesis": None,
        "verdict": None,
        "citations": [],
    }


# ── Router ────────────────────────────────────────────────────────────────────

def _should_continue(state: AgentState) -> str:
    if state.get("error") or not state.get("filtered_papers"):
        return "error"
    return "continue"


# ── Compile graph ─────────────────────────────────────────────────────────────

_builder = StateGraph(AgentState)
_builder.add_node("query_agent",     orchestrate_query)
_builder.add_node("retrieval_agent", orchestrate_retrieval)
_builder.add_node("parallel_agents", orchestrate_parallel)
_builder.add_node("synthesis_agent", orchestrate_synthesis)
_builder.add_node("error_handler",   error_handler)

_builder.add_edge(START,             "query_agent")
_builder.add_edge("query_agent",     "retrieval_agent")
_builder.add_conditional_edges(
    "retrieval_agent",
    _should_continue,
    {"continue": "parallel_agents", "error": "error_handler"},
)
_builder.add_edge("parallel_agents", "synthesis_agent")
_builder.add_edge("synthesis_agent", END)
_builder.add_edge("error_handler",   END)

graph = _builder.compile()


# ── Public entry point ────────────────────────────────────────────────────────

def run_agent(clinical_question: str, mode: str = "scout") -> AgentState:
    """Run the full multi-agent pipeline and return final state."""
    initial: AgentState = {
        "clinical_question": clinical_question,
        "mode": mode,
        "messages": [],
        "search_strategies": {},
        "search_queries": [],
        "raw_papers": [],
        "filtered_papers": [],
        "quality_scores": [],
        "findings": [],
        "contradictions": [],
        "guideline_conflicts": [],
        "synthesis": None,
        "citations": [],
        "verdict": None,
        "reasoning_trace": [],
        "step_count": 0,
        "error": None,
        "retry_count": 0,
    }
    return graph.invoke(initial)
