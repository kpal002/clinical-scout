"""Agent state schema — Clinical Literature Scout + Medical Myth Debunker."""
from __future__ import annotations

from typing import Dict, List, Optional

try:
    from typing import Annotated, TypedDict
except ImportError:
    from typing_extensions import Annotated, TypedDict  # Python < 3.11

from langgraph.graph.message import add_messages


class Paper(dict):
    """A single PubMed paper with metadata and optional σ-RAG relevance score."""


class StudyQuality(TypedDict):
    """Evidence-hierarchy score for one paper."""
    pmid: str
    design: str    # e.g. "Randomized Controlled Trial"
    level: int     # 1 (lowest) – 5 (highest)
    confidence: str  # "high" | "moderate" | "low"


class Finding(TypedDict):
    """A single extracted claim with supporting paper IDs."""
    claim: str
    pmids: List[str]
    direction: str  # "positive" | "negative" | "neutral"


class Contradiction(TypedDict):
    """Two papers that reach conflicting conclusions on the same topic."""
    topic: str
    finding_a: str
    pmid_a: str
    finding_b: str
    pmid_b: str


class AgentState(dict):
    """Full mutable state threaded through the multi-agent pipeline."""

    # ── Input ─────────────────────────────────────────────────────────────────
    clinical_question: str
    mode: str  # "scout" | "debunker"

    # ── Conversation history ───────────────────────────────────────────────────
    messages: Annotated[list, add_messages]

    # ── Query Agent ────────────────────────────────────────────────────────────
    # strategy_name → list of PubMed query strings
    search_strategies: Dict[str, List[str]]
    search_queries: List[str]  # flat list (backward-compat)

    # ── Retrieval Agent ────────────────────────────────────────────────────────
    raw_papers: List[Paper]
    filtered_papers: List[Paper]

    # ── Quality Agent (parallel with Analysis) ─────────────────────────────────
    quality_scores: List[StudyQuality]

    # ── Analysis Agent (parallel with Quality) ─────────────────────────────────
    findings: List[Finding]
    contradictions: List[Contradiction]

    # ── Synthesis Agent ────────────────────────────────────────────────────────
    synthesis: Optional[str]
    citations: List[str]
    verdict: Optional[str]  # "BUSTED" | "SUPPORTED" | "MIXED" (debunker only)

    # ── Control ────────────────────────────────────────────────────────────────
    reasoning_trace: List[str]
    step_count: int
    error: Optional[str]
    retry_count: int
