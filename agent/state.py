"""Agent state schema for the Clinical Literature Scout + Medical Myth Debunker."""
from typing import List, Optional

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated  # Python < 3.9

from langgraph.graph.message import add_messages


class Paper(dict):
    """A single PubMed paper with metadata and optional σ-RAG relevance score."""


class AgentState(dict):
    """Full mutable state threaded through the LangGraph pipeline."""

    # Input
    clinical_question: str

    # Conversation history
    messages: Annotated[list, add_messages]

    # Intermediate results
    search_queries: List[str]
    raw_papers: List[Paper]
    filtered_papers: List[Paper]

    # Mode
    mode: str  # "scout" or "debunker"
    verdict: Optional[str]  # "BUSTED" | "SUPPORTED" | "MIXED" (debunker only)

    # Output
    synthesis: Optional[str]
    citations: List[str]
    reasoning_trace: List[str]

    # Control
    step_count: int
    error: Optional[str]
    retry_count: int
