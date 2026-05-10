#!/usr/bin/env python3
"""Hugging Face Spaces demo — Clinical Literature Scout + Medical Myth Debunker.

Shows the agent making decisions step-by-step:
  1. Formulate PubMed queries  (Claude)
  2. Search PubMed             (NCBI E-utilities)
  3. Fetch abstracts           (NCBI E-utilities)
  4. σ-RAG significance filter (sigma-rag)
  5. Evidence synthesis        (Claude)
"""
from __future__ import annotations

import json
import os
import sys
from typing import Generator, List, Optional

import anthropic
import gradio as gr

# ── Path bootstrap ───────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from retrieval.pubmed import fetch_abstracts, search_pubmed  # noqa: E402
from retrieval.sigma_filter import filter_papers  # noqa: E402
from agent.state import Paper  # noqa: E402

# ── Config ───────────────────────────────────────────────────────────────────
_MODEL = "claude-sonnet-4-5"
_SIGMA_THRESHOLD = 1.2
_SIGMA_RETRY = 0.7
_MAX_PER_QUERY = 20
_MAX_FILTERED = 10

_QUERY_SYSTEM = {
    "scout": (
        "You are a biomedical librarian expert in PubMed MeSH query formulation. "
        "Given a clinical question, produce exactly 3 optimised PubMed search queries "
        "that together cover the topic broadly. Prioritise meta-analyses, systematic "
        "reviews, and RCTs. Return ONLY a JSON array of 3 strings. No other text."
    ),
    "debunker": (
        "You are a biomedical librarian. Given a medical myth or claim, produce exactly "
        "3 PubMed search queries that will find scientific evidence about this topic. "
        "Return ONLY a JSON array of 3 strings. No other text."
    ),
}

_SYNTHESIS_SYSTEM = {
    "scout": (
        "You are a clinical evidence synthesizer. Given a clinical question and filtered "
        "abstracts, produce a structured synthesis:\n\n"
        "## ANSWER\n(Direct answer, 2–3 sentences)\n\n"
        "## STRENGTH OF EVIDENCE\n(Strong/Moderate/Weak/Insufficient) — one sentence\n\n"
        "## KEY FINDINGS\n- Bullet points, each citing [N]\n\n"
        "## LIMITATIONS\n- Key gaps or contradictions\n\n"
        "## CITATIONS\n[N] Authors (Year). Title. Journal.\n\n"
        "Cite papers for every specific claim. Never fabricate."
    ),
    "debunker": (
        "You are a medical myth debunker. Given a myth and filtered abstracts, produce:\n\n"
        "## VERDICT: BUSTED / SUPPORTED / MIXED EVIDENCE\n\n"
        "## WHAT THE EVIDENCE SAYS\n(2–3 sentences)\n\n"
        "## KEY STUDIES\n- Bullet points, each citing [N]\n\n"
        "## COMMON MISCONCEPTION\n(Why people believe it — 1–2 sentences)\n\n"
        "## CITATIONS\n[N] Authors (Year). Title. Journal.\n\n"
        "Be direct. Lead with the verdict. Write for a general audience."
    ),
}

_VERDICT_EMOJI = {"BUSTED": "🟢", "SUPPORTED": "🔴", "MIXED": "🟡"}

_EXAMPLES = [
    ["What is the evidence for GLP-1 receptor agonists in reducing cardiovascular risk?", "scout"],
    ["Does metformin reduce cancer risk in diabetic patients?", "scout"],
    ["Does cracking your knuckles cause arthritis?", "debunker"],
    ["Do we only use 10% of our brain?", "debunker"],
    ["Does vitamin C prevent the common cold?", "debunker"],
]


# ── Pipeline helpers ─────────────────────────────────────────────────────────

def _client() -> anthropic.Anthropic:
    """Return an Anthropic client or raise if key is missing."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY is not set in Space secrets.")
    return anthropic.Anthropic(api_key=key)


def _formulate_queries(question: str, mode: str) -> List[str]:
    """Ask Claude to produce 3 optimised PubMed queries."""
    resp = _client().messages.create(
        model=_MODEL,
        max_tokens=512,
        system=_QUERY_SYSTEM[mode],
        messages=[{"role": "user", "content": question}],
    )
    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(raw)


def _synthesize(papers: List[Paper], question: str, mode: str) -> str:
    """Ask Claude to synthesize evidence from filtered papers."""
    label = "Medical Myth" if mode == "debunker" else "Clinical Question"
    numbered = "\n\n".join(
        f"[{i + 1}] {p['authors']} ({p['year']}) — {p['journal']}\n"
        f"Title: {p['title']}\n"
        f"Abstract: {p['abstract'][:1200]}"
        for i, p in enumerate(papers)
    )
    resp = _client().messages.create(
        model=_MODEL,
        max_tokens=2048,
        system=_SYNTHESIS_SYSTEM[mode],
        messages=[{
            "role": "user",
            "content": f"{label}: {question}\n\nPapers:\n{numbered}",
        }],
    )
    return resp.content[0].text.strip()


def _extract_verdict(synthesis: str) -> Optional[str]:
    """Scan the synthesis for the first VERDICT keyword."""
    for line in synthesis.splitlines():
        upper = line.upper()
        if "BUSTED" in upper:
            return "BUSTED"
        if "SUPPORTED" in upper:
            return "SUPPORTED"
        if "MIXED" in upper:
            return "MIXED"
    return None


# ── Streaming pipeline ────────────────────────────────────────────────────────

def stream_pipeline(
    question: str, mode: str
) -> Generator[tuple[str, str, str], None, None]:
    """Run the full pipeline and yield (log, synthesis, citations) after each step."""

    log = ""
    synthesis = ""
    citations = ""

    def add(text: str = "") -> None:
        nonlocal log
        log += text + "\n"

    # Guard
    if not question.strip():
        yield "⚠️ Please enter a question.", synthesis, citations
        return

    try:
        _client()
    except ValueError as exc:
        yield f"❌ **{exc}**\n\nAdd `ANTHROPIC_API_KEY` in **Settings → Secrets**.", "", ""
        return

    # ── Step 1 ──────────────────────────────────────────────────────────────
    add("## 🔍 Step 1 — Formulating PubMed Search Queries")
    add("> Asking Claude to translate the question into optimised MeSH queries…")
    yield log, synthesis, citations

    try:
        queries = _formulate_queries(question, mode)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        add(f"\n❌ Query formulation failed: `{exc}`")
        yield log, synthesis, citations
        return

    for i, q in enumerate(queries, 1):
        add(f"- **Query {i}:** `{q}`")
    add("")
    yield log, synthesis, citations

    # ── Step 2 ──────────────────────────────────────────────────────────────
    add("---\n## 📡 Step 2 — Searching PubMed")
    yield log, synthesis, citations

    all_pmids: set[str] = set()
    for q in queries:
        try:
            pmids = search_pubmed(q, max_results=_MAX_PER_QUERY)
        except Exception:  # pylint: disable=broad-exception-caught
            pmids = []
        short_q = q[:60] + "…" if len(q) > 60 else q
        add(f"- `{short_q}` → **{len(pmids)} results**")
        all_pmids.update(pmids)
        yield log, synthesis, citations

    unique_pmids = list(all_pmids)
    add(f"\n**Total unique PMIDs collected: {len(unique_pmids)}**\n")
    yield log, synthesis, citations

    if not unique_pmids:
        add("⚠️ No results found. Try rephrasing your question.")
        yield log, synthesis, citations
        return

    # ── Step 3 ──────────────────────────────────────────────────────────────
    add("---\n## 📄 Step 3 — Fetching Abstracts from NCBI")
    yield log, synthesis, citations

    try:
        papers = fetch_abstracts(unique_pmids)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        add(f"\n❌ Abstract fetch failed: `{exc}`")
        yield log, synthesis, citations
        return

    no_abstract = len(unique_pmids) - len(papers)
    add(f"Fetched **{len(papers)} abstracts** *(+{no_abstract} had none)*\n")
    yield log, synthesis, citations

    # ── Step 4 ──────────────────────────────────────────────────────────────
    threshold = _SIGMA_THRESHOLD
    add("---\n## 🔬 Step 4 — σ-RAG Significance Filter")
    add(
        f"Fitting a Gaussian noise floor on the **{len(papers)}-paper corpus**, "
        f"then keeping only papers that score above **{threshold}σ** above background."
    )
    add("\n| Paper | σ score | Pass? |")
    add("|-------|---------|-------|")
    yield log, synthesis, citations

    rows: List[str] = []

    def _cb(title: str, score: float, passed: bool) -> None:
        icon = "✅" if passed else "❌"
        short = (title[:65] + "…") if len(title) > 65 else title
        rows.append(f"| {short} | `{score:.2f}σ` | {icon} |")

    filtered = filter_papers(
        papers, question,
        sigma_threshold=threshold,
        max_results=_MAX_FILTERED,
        progress_callback=_cb,
    )

    # Retry at lower threshold if nothing passed
    if not filtered and papers:
        rows.clear()
        threshold = _SIGMA_RETRY
        add(f"\n> Nothing passed {_SIGMA_THRESHOLD}σ — retrying at **{threshold}σ**…\n")
        add("| Paper | σ score | Pass? |")
        add("|-------|---------|-------|")
        yield log, synthesis, citations
        filtered = filter_papers(
            papers, question,
            sigma_threshold=threshold,
            max_results=_MAX_FILTERED,
            progress_callback=_cb,
        )

    log += "\n".join(rows) + "\n\n"
    add(
        f"**{len(papers)} retrieved → "
        f"{len(filtered)} passed σ filter → "
        f"{len(papers) - len(filtered)} filtered out**\n"
    )
    yield log, synthesis, citations

    if not filtered:
        add("⚠️ No papers cleared the significance threshold. Try rephrasing.")
        yield log, synthesis, citations
        return

    # ── Step 5 ──────────────────────────────────────────────────────────────
    add("---\n## 🧠 Step 5 — Evidence Synthesis")
    add(f"> Sending {len(filtered)} high-signal abstracts to Claude…")
    yield log, synthesis, citations

    try:
        synthesis = _synthesize(filtered, question, mode)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        add(f"\n❌ Synthesis failed: `{exc}`")
        yield log, synthesis, citations
        return

    if mode == "debunker":
        verdict = _extract_verdict(synthesis)
        if verdict:
            emoji = _VERDICT_EMOJI.get(verdict, "⚪")
            add(f"\n### {emoji} Verdict: **{verdict}**")

    add("\n---\n✅ **Pipeline complete.**")
    yield log, synthesis, citations

    citations = "\n\n".join(
        f"**[{i + 1}]** {p['authors']} ({p['year']}). "
        f"*{p['title']}*. {p['journal']}. "
        f"[PubMed {p['pmid']}]({p['url']})"
        for i, p in enumerate(filtered)
    )
    yield log, synthesis, citations


# ── Gradio UI ─────────────────────────────────────────────────────────────────

_TITLE = """
# 🔬 Clinical Literature Scout + Medical Myth Debunker
**Multi-step AI agent — powered by σ-RAG + PubMed + Claude**

Watch the agent reason in real time: it formulates queries, searches PubMed,
applies significance-threshold filtering (σ-RAG), then synthesizes evidence.
"""

_HOW_IT_WORKS = """
### How it works

| Step | Tool | What happens |
|------|------|-------------|
| 1 | **Claude** | Translates your question into 3 optimised PubMed MeSH queries |
| 2 | **NCBI Esearch** | Searches PubMed and collects unique paper IDs |
| 3 | **NCBI Efetch** | Downloads full abstracts for each paper |
| 4 | **σ-RAG** | Fits a Gaussian noise floor; keeps only papers above the significance threshold |
| 5 | **Claude** | Synthesizes a structured evidence report with citations |

σ-RAG eliminates noise by treating retrieval as a signal-detection problem —
only papers whose relevance score is statistically above background are passed
to the LLM, preventing hallucination from irrelevant context.
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan"),
    title="Clinical Literature Scout",
) as demo:

    gr.Markdown(_TITLE)

    with gr.Row():
        mode_radio = gr.Radio(
            choices=["scout", "debunker"],
            value="scout",
            label="Mode",
            info="Scout = clinical evidence  |  Debunker = myth verification",
            scale=1,
        )
        question_box = gr.Textbox(
            label="Your question or myth",
            placeholder=(
                "E.g. What is the evidence for GLP-1 receptor agonists "
                "in reducing cardiovascular risk?"
            ),
            lines=2,
            scale=4,
        )

    run_btn = gr.Button("🚀 Run Agent", variant="primary", size="lg")

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### 🤖 Agent Reasoning Log")
            log_out = gr.Markdown(
                value="*Press **Run Agent** to start. The agent log will stream here.*"
            )
        with gr.Column(scale=2):
            gr.Markdown("### 📋 Evidence Synthesis")
            synthesis_out = gr.Markdown(
                value="*Synthesis will appear here once the agent finishes Step 5.*"
            )

    with gr.Accordion("📚 Citations", open=False):
        citations_out = gr.Markdown(value="*Citations will appear after synthesis.*")

    gr.Markdown("### 💡 Try an example")
    with gr.Row():
        for ex_q, ex_m in _EXAMPLES:
            short = ex_q[:55] + "…" if len(ex_q) > 55 else ex_q
            gr.Button(short, size="sm").click(
                fn=lambda q=ex_q, m=ex_m: (q, m),
                outputs=[question_box, mode_radio],
            )

    with gr.Accordion("ℹ️ How it works", open=False):
        gr.Markdown(_HOW_IT_WORKS)

    run_btn.click(
        fn=stream_pipeline,
        inputs=[question_box, mode_radio],
        outputs=[log_out, synthesis_out, citations_out],
        show_progress="hidden",
    )

if __name__ == "__main__":
    demo.launch()
