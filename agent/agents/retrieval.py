"""Retrieval Agent — parallel PubMed search + σ-RAG filtering per strategy."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

from agent.state import Paper
from retrieval import pubmed, sigma_filter

_MAX_PER_QUERY = 20
_MAX_FILTERED = 12
_SIGMA_THRESHOLD = 1.2
_SIGMA_RETRY = 0.7


def _search_strategy(queries: List[str]) -> List[Paper]:
    """Search PubMed for all queries in one strategy, fetch abstracts, return papers."""
    pmids: set[str] = set()
    for q in queries:
        try:
            pmids.update(pubmed.search_pubmed(q, max_results=_MAX_PER_QUERY))
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    if not pmids:
        return []
    papers = pubmed.fetch_abstracts(list(pmids))
    return papers


def run(
    strategies: Dict[str, List[str]],
    question: str,
) -> Tuple[List[Paper], List[Paper]]:
    """Run 3 strategy searches in parallel, merge, deduplicate, then σ-RAG filter.

    Returns (raw_papers, filtered_papers).
    """
    # ── Parallel searches ──────────────────────────────────────────────────────
    all_papers: Dict[str, Paper] = {}  # pmid → Paper (deduplication)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(_search_strategy, queries): name
            for name, queries in strategies.items()
            if queries
        }
        for future in as_completed(futures):
            try:
                for paper in future.result():
                    all_papers.setdefault(paper["pmid"], paper)
            except Exception:  # pylint: disable=broad-exception-caught
                pass

    raw = list(all_papers.values())

    # ── σ-RAG filtering ────────────────────────────────────────────────────────
    filtered = sigma_filter.filter_papers(
        raw, question,
        sigma_threshold=_SIGMA_THRESHOLD,
        max_results=_MAX_FILTERED,
    )
    if not filtered and raw:
        filtered = sigma_filter.filter_papers(
            raw, question,
            sigma_threshold=_SIGMA_RETRY,
            max_results=_MAX_FILTERED,
        )

    return raw, filtered
