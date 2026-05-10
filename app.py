#!/usr/bin/env python3
"""Hugging Face Spaces demo — Clinical Literature Scout + Medical Myth Debunker.

Multi-agent pipeline streamed live:
  Orchestrator
  ├── Query Agent        → 3 specialised search strategies (parallel)
  ├── Retrieval Agent    → parallel PubMed searches + σ-RAG filter
  ├── Quality Agent  ┐   → study design scoring          (parallel)
  ├── Analysis Agent ┘   → findings + contradiction detection
  └── Synthesis Agent    → claim-level citations
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Generator, List, Tuple

import anthropic
import gradio as gr

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from agent.agents import analysis, quality, query, synthesis  # noqa: E402  # pylint: disable=wrong-import-position
from agent.agents.synthesis import SynthesisInput  # noqa: E402  # pylint: disable=wrong-import-position
from agent.state import Contradiction, Finding, Paper, StudyQuality  # noqa: E402  # pylint: disable=wrong-import-position
from retrieval import pubmed as _pubmed, sigma_filter as _sf  # noqa: E402  # pylint: disable=wrong-import-position

_EXAMPLES = [
    ["What is the evidence for GLP-1 receptor agonists in reducing cardiovascular risk?", "scout"],
    ["Does metformin reduce cancer risk in diabetic patients?", "scout"],
    ["Does cracking your knuckles cause arthritis?", "debunker"],
    ["Do we only use 10% of our brain?", "debunker"],
    ["Does vitamin C prevent the common cold?", "debunker"],
]

_VERDICT_COLOR = {"BUSTED": "#22c55e", "SUPPORTED": "#ef4444", "MIXED": "#f59e0b"}
_VERDICT_LABEL = {"BUSTED": "BUSTED ✓", "SUPPORTED": "SUPPORTED", "MIXED": "MIXED EVIDENCE"}

# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
* { box-sizing: border-box; }
body, .gradio-container {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    background: #0a0a0f !important;
}
.gradio-container { max-width: 1400px !important; margin: 0 auto !important; }

.app-header { padding: 40px 0 28px; text-align: center; }
.app-header h1 {
    font-size: 2rem; font-weight: 700; margin: 0 0 8px;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #34d399 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    letter-spacing: -0.5px;
}
.app-header p { color: #4b5563; font-size: 0.85rem; margin: 0; }

.input-card {
    background: #111118; border: 1px solid #1e1e2e;
    border-radius: 16px; padding: 20px; margin-bottom: 14px;
}
.mode-radio label { color: #9ca3af !important; font-size: 0.78rem !important; }
.mode-radio .wrap { gap: 6px !important; }
.mode-radio .wrap label {
    background: #1a1a2e !important; border: 1px solid #2a2a3e !important;
    border-radius: 8px !important; padding: 5px 14px !important;
    color: #9ca3af !important; font-size: 0.82rem !important;
    font-weight: 500 !important; cursor: pointer !important; transition: all 0.2s !important;
}
.mode-radio .wrap label:has(input:checked) {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    border-color: transparent !important; color: white !important;
}
.question-box textarea {
    background: #0d0d1a !important; border: 1px solid #2a2a3e !important;
    border-radius: 10px !important; color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.93rem !important; resize: none !important;
}
.question-box textarea:focus {
    border-color: #3b82f6 !important; box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important;
}
.run-btn {
    background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%) !important;
    border: none !important; border-radius: 12px !important; color: white !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.95rem !important;
    font-weight: 600 !important; height: 50px !important; letter-spacing: 0.3px !important;
    transition: all 0.2s !important; box-shadow: 0 4px 20px rgba(99,102,241,0.3) !important;
}
.run-btn:hover { transform: translateY(-1px) !important; box-shadow: 0 8px 28px rgba(99,102,241,0.45) !important; }

.panel-label {
    color: #374151; font-size: 0.68rem; font-weight: 600;
    letter-spacing: 1.5px; text-transform: uppercase;
    margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid #1a1a2a;
}
.log-panel {
    background: #0d0d18; border: 1px solid #1a1a2a;
    border-radius: 16px; padding: 22px; min-height: 560px;
}
.synth-panel {
    background: #0d0d18; border: 1px solid #1a1a2a;
    border-radius: 16px; padding: 22px; min-height: 560px;
}
.synth-panel h2, .synth-panel h3 { color: #e2e8f0 !important; font-weight: 600 !important; }
.synth-panel p, .synth-panel li { color: #94a3b8 !important; line-height: 1.7 !important; }
.citations-accordion {
    background: #0d0d18 !important; border: 1px solid #1a1a2a !important;
    border-radius: 12px !important; margin-top: 10px;
}
.example-btn button {
    background: #111118 !important; border: 1px solid #1e1e2e !important;
    border-radius: 8px !important; color: #6b7280 !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.75rem !important;
    transition: all 0.2s !important; white-space: normal !important;
    text-align: left !important; height: auto !important; padding: 7px 11px !important;
}
.example-btn button:hover {
    background: #1a1a2e !important; border-color: #3b82f6 !important; color: #60a5fa !important;
}
.hide-label > label { display: none !important; }
footer { display: none !important; }

/* ── Agent tree styles ─────────────────────────────────────────────────── */
.agent-tree { display: flex; flex-direction: column; gap: 0; }

.orchestrator-header {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 14px; margin-bottom: 12px;
    background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(168,85,247,0.05));
    border: 1px solid rgba(99,102,241,0.2); border-radius: 10px;
    font-size: 0.82rem; font-weight: 600; color: #a78bfa;
}
.orch-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #6366f1; box-shadow: 0 0 8px #6366f1;
    animation: orch-pulse 2s ease-in-out infinite;
}
@keyframes orch-pulse {
    0%, 100% { opacity: 1; } 50% { opacity: 0.4; }
}

.agent-row { display: flex; gap: 0; position: relative; padding-left: 20px; }
.agent-row::before {
    content: '';
    position: absolute; left: 8px; top: 0; bottom: 0;
    width: 1px; background: #1e1e2e;
}
.agent-row:last-child::before { bottom: 50%; }

.branch-line {
    position: absolute; left: 8px; top: 50%; width: 12px; height: 1px;
    background: #2a2a3e;
}

.agent-card {
    flex: 1; margin: 4px 0 4px 12px;
    background: #111118; border: 1px solid #1e1e2e;
    border-radius: 10px; padding: 10px 14px;
    transition: border-color 0.3s;
}
.agent-card.active { border-color: #3b82f6; box-shadow: 0 0 12px rgba(59,130,246,0.12); }
.agent-card.done   { border-color: #1a3a2a; }
.agent-card.error  { border-color: #7f1d1d; }

.agent-card-header {
    display: flex; align-items: center; gap: 8px;
    font-size: 0.8rem; font-weight: 600; margin-bottom: 6px;
}
.agent-card-header.pending { color: #374151; }
.agent-card-header.active  { color: #60a5fa; }
.agent-card-header.done    { color: #34d399; }
.agent-card-header.error   { color: #ef4444; }

.status-dot {
    width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0;
}
.status-dot.pending { background: #1f2937; border: 1px solid #374151; }
.status-dot.active  { background: #3b82f6; box-shadow: 0 0 6px #3b82f6; animation: blink 1s ease-in-out infinite; }
.status-dot.done    { background: #22c55e; }
.status-dot.error   { background: #ef4444; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

.thinking {
    display: inline-flex; gap: 3px; align-items: center; margin-left: 4px;
}
.thinking span {
    width: 3px; height: 3px; border-radius: 50%; background: #6366f1;
    animation: bounce 1.1s ease-in-out infinite;
}
.thinking span:nth-child(2) { animation-delay: 0.18s; }
.thinking span:nth-child(3) { animation-delay: 0.36s; }
@keyframes bounce { 0%,80%,100%{transform:translateY(0);opacity:.4} 40%{transform:translateY(-4px);opacity:1} }

.agent-detail {
    font-size: 0.74rem; color: #4b5563; line-height: 1.6;
    padding-left: 15px;
}
.agent-detail.visible { color: #6b7280; }

/* parallel bracket */
.parallel-group {
    margin: 4px 0 4px 12px; padding-left: 12px;
    border-left: 2px solid #1e3a5f;
    display: flex; flex-direction: column; gap: 4px;
    position: relative;
}
.parallel-label {
    font-size: 0.65rem; font-weight: 600; color: #1e3a5f;
    letter-spacing: 1px; text-transform: uppercase;
    margin-bottom: 2px;
}

/* inline tags */
.tag {
    display: inline-block; padding: 1px 7px;
    border-radius: 4px; font-size: 0.68rem; font-weight: 500;
    font-family: 'JetBrains Mono', monospace;
}
.tag-blue  { background: #0f172a; border: 1px solid #1e3a5f; color: #7dd3fc; }
.tag-green { background: #052e16; border: 1px solid #14532d; color: #4ade80; }
.tag-amber { background: #1c1003; border: 1px solid #713f12; color: #fbbf24; }
.tag-gray  { background: #111118; border: 1px solid #2a2a3e; color: #6b7280; }

.sigma-table {
    width: 100%; border-collapse: collapse; margin-top: 6px; font-size: 0.72rem;
}
.sigma-table th {
    color: #374151; font-weight: 500; text-align: left; padding: 3px 7px;
    border-bottom: 1px solid #1e1e2e; font-size: 0.65rem;
    letter-spacing: 0.5px; text-transform: uppercase;
}
.sigma-table td { padding: 4px 7px; color: #6b7280; border-bottom: 1px solid #0d0d18; }
.sigma-table td.pass { color: #34d399; }
.sigma-table td.fail { color: #374151; }
.mono { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; }

.finding-row { display: flex; gap: 6px; align-items: flex-start; margin: 3px 0; font-size: 0.75rem; }
.finding-dir { flex-shrink: 0; font-size: 0.7rem; }
.contradiction-row {
    background: rgba(245,158,11,0.05); border: 1px solid rgba(245,158,11,0.15);
    border-radius: 6px; padding: 5px 8px; margin: 3px 0; font-size: 0.73rem; color: #92400e;
}

.complete-banner {
    background: linear-gradient(135deg,rgba(5,150,105,.08),rgba(52,211,153,.04));
    border: 1px solid rgba(52,211,153,.18); border-radius: 8px;
    padding: 10px 14px; margin-top: 14px;
    color: #34d399; font-size: 0.78rem; font-weight: 500;
    display: flex; align-items: center; gap: 8px;
}
.empty-state {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; height: 300px; color: #1f2937; gap: 10px; text-align: center;
}
.empty-state .icon { font-size: 2rem; opacity: .25; }
.empty-state p { font-size: 0.8rem; margin: 0; }
"""

# ── HTML rendering helpers ────────────────────────────────────────────────────

def _thinking() -> str:
    return '<span class="thinking"><span></span><span></span><span></span></span>'


def _tag(text: str, kind: str = "gray") -> str:
    return f'<span class="tag tag-{kind}">{text}</span>'


def _status_dot(state: str) -> str:
    return f'<span class="status-dot {state}"></span>'


def _agent_card(
    icon: str,
    name: str,
    state: str,  # "pending" | "active" | "done" | "error"
    detail: str = "",
    thinking: bool = False,
) -> str:
    suffix = _thinking() if thinking else ""
    detail_html = (
        f'<div class="agent-detail visible">{detail}</div>' if detail else ""
    )
    return (
        f'<div class="agent-card {state}">'
        f'  <div class="agent-card-header {state}">'
        f'    {_status_dot(state)} {icon} {name}{suffix}'
        f'  </div>'
        f'  {detail_html}'
        f'</div>'
    )


def _render(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    q_state: str, q_detail: str,
    r_state: str, r_detail: str,
    qa_state: str, qa_detail: str,
    an_state: str, an_detail: str,
    sy_state: str, sy_detail: str,
    banner: str = "",
) -> str:
    parallel_block = (
        f'<div class="agent-row">'
        f'  <div class="branch-line"></div>'
        f'  <div class="parallel-group">'
        f'    <div class="parallel-label">⚡ parallel</div>'
        f'    {_agent_card("📊", "Quality Agent", qa_state, qa_detail, qa_state == "active")}'
        f'    {_agent_card("🔬", "Analysis Agent", an_state, an_detail, an_state == "active")}'
        f'  </div>'
        f'</div>'
    )
    return (
        '<div class="agent-tree">'
        '  <div class="orchestrator-header">'
        '    <div class="orch-dot"></div> Orchestrator Agent'
        '  </div>'
        f'  <div class="agent-row"><div class="branch-line"></div>'
        f'    {_agent_card("📝", "Query Agent", q_state, q_detail, q_state == "active")}'
        f'  </div>'
        f'  <div class="agent-row"><div class="branch-line"></div>'
        f'    {_agent_card("🔍", "Retrieval Agent", r_state, r_detail, r_state == "active")}'
        f'  </div>'
        f'  {parallel_block}'
        f'  <div class="agent-row"><div class="branch-line"></div>'
        f'    {_agent_card("🧠", "Synthesis Agent", sy_state, sy_detail, sy_state == "active")}'
        f'  </div>'
        f'  {banner}'
        f'</div>'
    )


def _sigma_row(title: str, score: float, passed: bool) -> str:
    short = (title[:62] + "…") if len(title) > 62 else title
    icon = "✓" if passed else "✗"
    td = "pass" if passed else "fail"
    score_cls = "mono pass" if passed else "mono fail"
    return (
        f"<tr>"
        f'<td class="{td}" style="max-width:340px;overflow:hidden;'
        f'text-overflow:ellipsis;white-space:nowrap">{short}</td>'
        f'<td><span class="{score_cls}">{score:.2f}σ</span></td>'
        f'<td class="{td}">{icon}</td>'
        f"</tr>"
    )


# ── Pipeline ──────────────────────────────────────────────────────────────────

def _get_client() -> anthropic.Anthropic:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY is not set in Space secrets.")
    return anthropic.Anthropic(api_key=key)


def _search_one_strategy(name: str, queries: list) -> Tuple[str, List[Paper]]:
    """Search PubMed for one strategy's queries and return (name, papers)."""
    pmids: set[str] = set()
    for q_ in queries:
        try:
            pmids.update(_pubmed.search_pubmed(q_, max_results=20))
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    papers_ = _pubmed.fetch_abstracts(list(pmids)) if pmids else []
    return name, papers_


def _sigma_filter(raw: List[Paper], question: str) -> Tuple[List[Paper], str]:
    """Run σ-RAG on raw papers; return (filtered_papers, sigma_html_table)."""
    sigma_rows = ""

    def _cb(title: str, score: float, passed: bool) -> None:
        nonlocal sigma_rows
        sigma_rows += _sigma_row(title, score, passed)

    filtered = _sf.filter_papers(raw, question, sigma_threshold=1.2, max_results=12,
                                  progress_callback=_cb)
    if not filtered and raw:
        sigma_rows = ""
        filtered = _sf.filter_papers(raw, question, sigma_threshold=0.7, max_results=12,
                                      progress_callback=_cb)

    table = (
        '<table class="sigma-table"><thead><tr>'
        '<th>Paper</th><th>σ score</th><th></th>'
        f'</tr></thead><tbody>{sigma_rows}</tbody></table>'
    )
    return filtered, table


def _run_retrieval(
    strategies: dict, question: str
) -> Tuple[List[Paper], List[Paper], str, str]:
    """Parallel PubMed search + σ-RAG. Returns (raw, filtered, r_detail, sigma_table)."""
    all_papers_map: dict[str, Paper] = {}
    r_detail_parts: list[str] = []

    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = {
            pool.submit(_search_one_strategy, n, qs): n
            for n, qs in strategies.items() if qs
        }
        for fut in as_completed(futs):
            name_, papers_ = fut.result()
            for p in papers_:
                all_papers_map.setdefault(p["pmid"], p)
            label = name_.replace("_", " ")
            r_detail_parts.append(
                f'{_tag(label, "blue")} → {_tag(str(len(papers_)) + " papers", "gray")}'
            )

    raw = list(all_papers_map.values())
    filtered, sigma_table = _sigma_filter(raw, question)
    return raw, filtered, "  ".join(r_detail_parts), sigma_table


def _verdict_badge(mode: str, verdict: str) -> str:
    """Return an HTML badge for the debunker verdict, or empty string."""
    if mode != "debunker" or not verdict:
        return ""
    color = _VERDICT_COLOR.get(verdict, "#6b7280")
    label = _VERDICT_LABEL.get(verdict, verdict)
    return (
        f'<span style="background:rgba(0,0,0,.4);border:1px solid {color};'
        f'color:{color};padding:2px 10px;border-radius:12px;font-size:0.72rem;'
        f'font-weight:700;margin-top:4px;display:inline-block">{label}</span>'
    )


def _citations_md(papers: List[Paper]) -> str:
    """Format filtered papers as markdown citation list."""
    return "\n\n".join(
        f"**[{i + 1}]** {p['authors']} ({p['year']}). "
        f"*{p['title']}*. {p['journal']}. "
        f"[PubMed {p['pmid']}]({p['url']})"
        for i, p in enumerate(papers)
    )


def _run_parallel_agents(
    filtered: List[Paper], question: str, mode: str, client: anthropic.Anthropic
) -> Tuple[List[StudyQuality], List[Finding], List[Contradiction]]:
    """Run Quality and Analysis agents concurrently."""
    quality_scores: List[StudyQuality] = []
    findings: List[Finding] = []
    contradictions: List[Contradiction] = []

    with ThreadPoolExecutor(max_workers=2) as pool:
        q_fut = pool.submit(quality.run, filtered)
        a_fut = pool.submit(analysis.run, filtered, question, mode, client)
        for fut in as_completed([q_fut, a_fut]):
            result = fut.result()
            if fut is q_fut:
                quality_scores = result
            else:
                findings, contradictions = result

    return quality_scores, findings, contradictions


def _qa_detail_html(quality_scores: List[StudyQuality]) -> str:
    lvl_counts: dict[int, int] = {}
    for s in quality_scores:
        lvl_counts[s["level"]] = lvl_counts.get(s["level"], 0) + 1
    lvl_labels = {
        5: ("meta-analyses", "green"), 4: ("RCTs", "blue"),
        3: ("cohort", "gray"), 2: ("case-ctrl", "gray"), 1: ("opinion", "gray"),
    }
    return "  ".join(
        _tag(f'{cnt} {lvl_labels.get(lvl, ("other", "gray"))[0]}',
             lvl_labels.get(lvl, ("other", "gray"))[1])
        for lvl, cnt in sorted(lvl_counts.items(), reverse=True)
    )


def _an_detail_html(findings: List[Finding], contradictions: List[Contradiction]) -> str:
    dir_icon = {"positive": "↑", "negative": "↓", "neutral": "→"}
    dir_color = {"positive": "#34d399", "negative": "#ef4444", "neutral": "#6b7280"}
    findings_html = "".join(
        f'<div class="finding-row">'
        f'<span class="finding-dir" style="color:{dir_color.get(f["direction"], "#6b7280")}">'
        f'{dir_icon.get(f["direction"], "→")}</span>'
        f'<span style="color:#6b7280">'
        f'{f["claim"][:90]}{"…" if len(f["claim"]) > 90 else ""}</span>'
        f'</div>'
        for f in findings[:5]
    )
    contra_html = "".join(
        f'<div class="contradiction-row">⚡ {c["topic"]}</div>'
        for c in contradictions[:3]
    )
    return findings_html + (contra_html if contradictions else "")


def stream_pipeline(
    question: str, mode: str
) -> Generator[Tuple[str, str, str], None, None]:
    """Run the multi-agent pipeline and yield (log_html, synthesis_md, citations_md)."""

    synthesis_md = ""
    citations_md = ""

    # Initial state — all pending
    def emit(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        q=("pending", ""), r=("pending", ""),
        qa=("pending", ""), an=("pending", ""),
        sy=("pending", ""), banner=""
    ) -> Tuple[str, str, str]:
        return (
            _render(q[0], q[1], r[0], r[1], qa[0], qa[1], an[0], an[1], sy[0], sy[1], banner),
            synthesis_md,
            citations_md,
        )

    if not question.strip():
        err = '<p style="color:#ef4444;font-size:0.82rem">⚠ Please enter a question.</p>'
        yield err, synthesis_md, citations_md
        return

    try:
        client = _get_client()
    except ValueError as exc:
        err = f'<p style="color:#ef4444;font-size:0.82rem">❌ {exc}</p>'
        yield err, "", ""
        return

    # ── Query Agent ────────────────────────────────────────────────────────────
    yield emit(q=("active", ""))

    try:
        strategies = query.run(question, mode, client)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        yield emit(q=("error", f"❌ {exc}"))
        return

    flat_queries = [q_ for qs in strategies.values() for q_ in qs]
    strategy_tags = " ".join(
        _tag(name.replace("_", " "), "blue")
        for name in strategies
    )
    q_detail = (
        f'{strategy_tags}<br>'
        + "  ".join(
            f'<span class="mono" style="color:#4b5563">{q_[:55]}…</span>'
            if len(q_) > 55 else f'<span class="mono" style="color:#4b5563">{q_}</span>'
            for q_ in flat_queries[:6]
        )
    )
    yield emit(q=("done", q_detail))

    # ── Retrieval Agent ────────────────────────────────────────────────────────
    yield emit(q=("done", q_detail), r=("active", ""))

    raw_papers, filtered, r_detail_base, sigma_table = _run_retrieval(strategies, question)
    r_detail_full = (
        r_detail_base
        + f'<br>{_tag(str(len(raw_papers)) + " raw", "gray")} → '
        + f'{_tag(str(len(filtered)) + " passed σ-RAG", "green")}'
        + sigma_table
    )
    r_final_state = "done" if filtered else "error"
    yield emit(q=("done", q_detail), r=(r_final_state, r_detail_full))

    if not filtered:
        no_pass_detail = (
            r_detail_full
            + '<br><span style="color:#f59e0b;font-size:0.72rem">'
            + '⚠ No papers cleared threshold.</span>'
        )
        yield emit(q=("done", q_detail), r=("error", no_pass_detail))
        return

    # ── Quality Agent ∥ Analysis Agent ────────────────────────────────────────
    yield emit(
        q=("done", q_detail), r=("done", r_detail_full),
        qa=("active", ""), an=("active", ""),
    )

    quality_scores, findings, contradictions = _run_parallel_agents(
        filtered, question, mode, client
    )
    qa_detail = _qa_detail_html(quality_scores)
    an_detail = _an_detail_html(findings, contradictions)

    yield emit(
        q=("done", q_detail), r=("done", r_detail_full),
        qa=("done", qa_detail), an=("done", an_detail),
        sy=("active", ""),
    )

    # ── Synthesis Agent ────────────────────────────────────────────────────────
    try:
        synth, verdict, _ = synthesis.run(
            SynthesisInput(
                papers=filtered, scores=quality_scores, findings=findings,
                contradictions=contradictions, question=question, mode=mode,
            ),
            client,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        yield emit(
            q=("done", q_detail), r=("done", r_detail_full),
            qa=("done", qa_detail), an=("done", an_detail),
            sy=("error", f"❌ {exc}"),
        )
        return

    badge = _verdict_badge(mode, verdict or "")
    sy_detail = (
        _tag(str(len(filtered)) + " papers synthesized", "green")
        + (f"  {badge}" if badge else "")
    )
    banner = '<div class="complete-banner">✦ All agents complete — results ready</div>'
    synthesis_md = synth
    citations_md = _citations_md(filtered)

    yield emit(
        q=("done", q_detail), r=("done", r_detail_full),
        qa=("done", qa_detail), an=("done", an_detail),
        sy=("done", sy_detail), banner=banner,
    )
    yield emit(
        q=("done", q_detail), r=("done", r_detail_full),
        qa=("done", qa_detail), an=("done", an_detail),
        sy=("done", sy_detail), banner=banner,
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────────

_EMPTY_LOG = """
<div class="empty-state">
  <div class="icon">🤖</div>
  <p>Multi-agent pipeline will stream here</p>
</div>"""

_EMPTY_SYNTH = """
<div class="empty-state">
  <div class="icon">📋</div>
  <p>Evidence synthesis will appear here</p>
</div>"""

with gr.Blocks(css=_CSS, title="Clinical Literature Scout") as demo:

    gr.HTML("""
    <div class="app-header">
      <h1>Clinical Literature Scout</h1>
      <p>Multi-agent AI system &nbsp;·&nbsp; Query · Retrieval · Quality · Analysis · Synthesis</p>
    </div>
    """)

    with gr.Group(elem_classes="input-card"):
        with gr.Row():
            mode_radio = gr.Radio(
                choices=["scout", "debunker"], value="scout", label="Mode",
                info="Scout = clinical evidence  ·  Debunker = myth verification",
                scale=1, elem_classes="mode-radio",
            )
            question_box = gr.Textbox(
                label="Your question or myth",
                placeholder=(
                    "E.g. What is the evidence for GLP-1 receptor agonists"
                    " in reducing cardiovascular risk?"
                ),
                lines=2, scale=4, elem_classes="question-box",
            )

    run_btn = gr.Button("Run Agent →", variant="primary", size="lg", elem_classes="run-btn")

    with gr.Row(equal_height=True):
        with gr.Column(scale=3):
            gr.HTML('<div class="panel-label">🤖 &nbsp;Agent Orchestration</div>')
            log_out = gr.HTML(value=_EMPTY_LOG, elem_classes="log-panel hide-label")
        with gr.Column(scale=2):
            gr.HTML('<div class="panel-label">📋 &nbsp;Evidence Synthesis</div>')
            synthesis_out = gr.Markdown(value=_EMPTY_SYNTH, elem_classes="synth-panel hide-label")

    with gr.Accordion("📚 Citations", open=False, elem_classes="citations-accordion"):
        citations_out = gr.Markdown(value="*Citations will appear after synthesis.*")

    gr.HTML('<div class="panel-label" style="margin-top:20px">💡 &nbsp;Try an example</div>')
    with gr.Row():
        for ex_q, ex_m in _EXAMPLES:
            btn_label = ex_q[:58] + "…" if len(ex_q) > 58 else ex_q
            gr.Button(btn_label, size="sm", elem_classes="example-btn").click(
                fn=lambda q=ex_q, m=ex_m: (q, m),
                outputs=[question_box, mode_radio],
            )

    run_btn.click(
        fn=stream_pipeline,
        inputs=[question_box, mode_radio],
        outputs=[log_out, synthesis_out, citations_out],
        show_progress="hidden",
    )

if __name__ == "__main__":
    demo.launch()
