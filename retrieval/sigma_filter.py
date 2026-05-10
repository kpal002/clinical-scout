"""σ-RAG significance filtering for clinical papers."""
from __future__ import annotations

import contextlib
import io
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

# Path bootstrap — must precede sigma_rag / agent imports when running from repo root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from sigma_rag import SigmaIndex  # pylint: disable=wrong-import-position
from agent.state import Paper  # pylint: disable=wrong-import-position

# σ-RAG requires at least 10 documents to fit a noise floor
_MIN_DOCS_FOR_SIGMA = 10


@dataclass
class _ScoringConfig:
    """Parameters used when scoring each paper against the query."""

    sigma_threshold: float
    use_sigma: bool
    noise_floor: object  # sigma_rag.NoiseFloor or None
    progress_callback: Optional[Callable]


def _calibrate_quietly(index: SigmaIndex) -> None:
    """Calibrate the σ-RAG noise floor while suppressing non-critical warnings."""
    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        index.calibrate()


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity between two embedding vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def _to_sigma(sim: float, noise_floor) -> float:
    """Convert a raw cosine similarity to σ units above background."""
    return (sim - noise_floor.mu_) / (noise_floor.sigma_ + 1e-10)


def _score_papers(
    papers: List[Paper],
    query_embed: np.ndarray,
    index: SigmaIndex,
    cfg: _ScoringConfig,
) -> List[Tuple[Paper, float]]:
    """Score each paper and return (paper_with_score, score) pairs above threshold."""
    scored: List[Tuple[Paper, float]] = []

    for paper in papers:
        doc_embed = index.embedder.embed(paper["abstract"])
        sim = _cosine_sim(query_embed, doc_embed)
        score = _to_sigma(sim, cfg.noise_floor) if cfg.use_sigma else sim * 10

        paper_with_score: Paper = {**paper, "sigma_score": round(score, 2)}
        passed = score >= cfg.sigma_threshold

        if cfg.progress_callback:
            cfg.progress_callback(paper["title"], score, passed)

        if passed:
            scored.append((paper_with_score, score))

    return scored


def filter_papers(
    papers: List[Paper],
    clinical_question: str,
    sigma_threshold: float = 1.5,
    max_results: int = 10,
    progress_callback: Optional[Callable] = None,
) -> List[Paper]:
    """Apply σ-RAG significance filtering to eliminate irrelevant papers.

    For corpora with ≥ 10 papers the noise floor is calibrated and scores are
    in standard-deviation units above background.  Smaller corpora fall back to
    scaled cosine similarity.

    Returns papers sorted by sigma_score descending, capped at max_results.
    progress_callback(title, sigma_score, passed) is called for each paper.
    """
    if not papers:
        return []

    abstracts = [p["abstract"] for p in papers]
    index = SigmaIndex(n_sigma=sigma_threshold)
    index.add_documents(abstracts)

    query_embed = index.embedder.embed(clinical_question)
    use_sigma = len(papers) >= _MIN_DOCS_FOR_SIGMA

    if use_sigma:
        _calibrate_quietly(index)

    cfg = _ScoringConfig(
        sigma_threshold=sigma_threshold,
        use_sigma=use_sigma,
        noise_floor=index.noise_floor if use_sigma else None,
        progress_callback=progress_callback,
    )

    scored = _score_papers(papers, query_embed, index, cfg)
    scored.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in scored[:max_results]]
