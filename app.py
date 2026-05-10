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

from agent.agents import analysis, guidelines, quality, query, synthesis  # noqa: E402  # pylint: disable=wrong-import-position
from agent.agents.synthesis import SynthesisInput  # noqa: E402  # pylint: disable=wrong-import-position
from agent.state import Contradiction, Finding, GuidelineConflict, Paper, StudyQuality  # noqa: E402  # pylint: disable=wrong-import-position
from retrieval import pubmed as _pubmed, sigma_filter as _sf  # noqa: E402  # pylint: disable=wrong-import-position

_EXAMPLES = [
    ["What is the evidence for GLP-1 receptor agonists in reducing cardiovascular risk?", "scout"],
    ["Does metformin reduce cancer risk in diabetic patients?", "scout"],
    ["Does cracking your knuckles cause arthritis?", "debunker"],
    ["Do we only use 10% of our brain?", "debunker"],
    ["Does vitamin C prevent the common cold?", "debunker"],
]

_VERDICT_COLOR = {"BUSTED": "#00ffaa", "SUPPORTED": "#ff4444", "MIXED": "#ffaa00"}
_VERDICT_LABEL = {"BUSTED": "BUSTED ✓", "SUPPORTED": "SUPPORTED", "MIXED": "MIXED EVIDENCE"}

# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');

* { box-sizing: border-box; }

body, .gradio-container {
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace !important;
    background: #0a0a0a !important;
    color: #cccccc !important;
}

.gradio-container { max-width: 1400px !important; margin: 0 auto !important; }

/* ── Header ── */
.app-header {
    padding: 28px 0 18px;
    text-align: left;
    border-bottom: 1px solid #1a1a1a;
    margin-bottom: 20px;
}
.app-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #00ffaa;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 0 0 6px;
    font-family: 'JetBrains Mono', monospace;
}
.app-title::before { content: '> '; color: #444; }
.app-subtitle {
    font-size: 0.72rem;
    color: #444;
    letter-spacing: 1px;
    margin: 0;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Input card ── */
.input-card {
    background: #0f0f0f !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 2px !important;
    padding: 16px 18px !important;
    margin-bottom: 12px !important;
}

/* ── Panel wrappers ── */
.panel-wrap {
    background: #0f0f0f;
    border: 1px solid #1e1e1e;
    border-radius: 2px;
    padding: 18px;
    min-height: 560px;
    display: flex;
    flex-direction: column;
}
.panel-hdr {
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #00ffaa;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1a1a1a;
    flex-shrink: 0;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Synthesis panel ── */
.synth-outer {
    background: #0f0f0f !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 2px !important;
    overflow: hidden !important;
    min-height: 560px !important;
}
.synth-inner { padding: 18px !important; }
.synth-inner h2 {
    color: #00ffaa !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border-left: 2px solid #00ffaa !important;
    padding-left: 10px !important;
    margin: 18px 0 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.synth-inner h3 {
    color: #888 !important;
    font-size: 0.68rem !important;
    font-weight: 500 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    margin: 12px 0 6px !important;
}
.synth-inner p, .synth-inner li {
    color: #aaaaaa !important;
    line-height: 1.75 !important;
    font-size: 0.82rem !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.synth-inner strong { color: #cccccc !important; }
.synth-inner code {
    background: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    color: #00ffaa !important;
    padding: 1px 5px !important;
    border-radius: 2px !important;
}

.synthesis-header {
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #00ffaa;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1a1a1a;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Agent tree ── */
.agent-tree { display: flex; flex-direction: column; gap: 0; }

.orchestrator-header {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 12px; margin-bottom: 10px;
    background: #111;
    border: 1px solid #00ffaa33;
    border-radius: 2px;
    font-size: 0.72rem; font-weight: 600; color: #00ffaa;
    letter-spacing: 1px;
    font-family: 'JetBrains Mono', monospace;
}
.orch-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #00ffaa; box-shadow: 0 0 8px #00ffaa88;
    animation: orch-pulse 2s ease-in-out infinite;
    flex-shrink: 0;
}
@keyframes orch-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

.agent-row { display: flex; gap: 0; position: relative; padding-left: 20px; }
.agent-row::before {
    content: '';
    position: absolute; left: 8px; top: 0; bottom: 0;
    width: 1px; background: #222;
}
.agent-row:last-child::before { bottom: 50%; }

.branch-line {
    position: absolute; left: 8px; top: 50%; width: 12px; height: 1px;
    background: #2a2a2a;
}

.agent-card {
    flex: 1; margin: 3px 0 3px 12px;
    background: #111; border: 1px solid #1e1e1e;
    border-radius: 2px; padding: 9px 12px;
    transition: border-color 0.2s;
}
.agent-card.active { border-color: #00ffaa55; box-shadow: 0 0 10px rgba(0,255,170,0.08); }
.agent-card.done   { border-color: #00aa5533; }
.agent-card.error  { border-color: #aa220033; }

.agent-card-header {
    display: flex; align-items: center; gap: 8px;
    font-size: 0.72rem; font-weight: 600; margin-bottom: 5px;
    font-family: 'JetBrains Mono', monospace; letter-spacing: 0.5px;
}
.agent-card-header.pending { color: #333; }
.agent-card-header.active  { color: #00ffaa; }
.agent-card-header.done    { color: #00cc88; }
.agent-card-header.error   { color: #ff4444; }

.status-dot {
    width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0;
}
.status-dot.pending { background: #222; border: 1px solid #333; }
.status-dot.active  { background: #00ffaa; box-shadow: 0 0 6px #00ffaa88; animation: blink 1s ease-in-out infinite; }
.status-dot.done    { background: #00cc88; }
.status-dot.error   { background: #ff4444; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

.thinking {
    display: inline-flex; gap: 3px; align-items: center; margin-left: 4px;
}
.thinking span {
    width: 3px; height: 3px; border-radius: 50%; background: #00ffaa;
    animation: bounce 1.1s ease-in-out infinite;
}
.thinking span:nth-child(2) { animation-delay: 0.18s; }
.thinking span:nth-child(3) { animation-delay: 0.36s; }
@keyframes bounce { 0%,80%,100%{transform:translateY(0);opacity:.3} 40%{transform:translateY(-4px);opacity:1} }

.agent-detail { font-size: 0.68rem; color: #444; line-height: 1.6; padding-left: 14px; font-family: 'JetBrains Mono', monospace; }
.agent-detail.visible { color: #666; }

/* parallel bracket */
.parallel-group {
    margin: 3px 0 3px 12px; padding-left: 12px;
    border-left: 1px solid #00ffaa22;
    display: flex; flex-direction: column; gap: 3px;
    position: relative;
}
.parallel-label {
    font-size: 0.58rem; font-weight: 600; color: #00ffaa44;
    letter-spacing: 2px; text-transform: uppercase; margin-bottom: 2px;
    font-family: 'JetBrains Mono', monospace;
}

/* inline tags */
.tag {
    display: inline-block; padding: 1px 7px;
    border-radius: 2px; font-size: 0.65rem; font-weight: 500;
    font-family: 'JetBrains Mono', monospace; letter-spacing: 0.3px;
}
.tag-blue  { background: #001a2a; border: 1px solid #003355; color: #44aadd; }
.tag-green { background: #001a0f; border: 1px solid #003322; color: #00ffaa; }
.tag-amber { background: #1a1200; border: 1px solid #332200; color: #ffaa00; }
.tag-gray  { background: #111; border: 1px solid #222; color: #555; }

.sigma-table {
    width: 100%; border-collapse: collapse; margin-top: 6px; font-size: 0.67rem;
    font-family: 'JetBrains Mono', monospace;
}
.sigma-table th {
    color: #333; font-weight: 500; text-align: left; padding: 3px 7px;
    border-bottom: 1px solid #1e1e1e; font-size: 0.6rem;
    letter-spacing: 1px; text-transform: uppercase;
}
.sigma-table td { padding: 3px 7px; color: #444; border-bottom: 1px solid #0f0f0f; }
.sigma-table td.pass { color: #00cc88; }
.sigma-table td.fail { color: #2a2a2a; }
.mono { font-family: 'JetBrains Mono', monospace; font-size: 0.66rem; }

.finding-row { display: flex; gap: 6px; align-items: flex-start; margin: 3px 0; font-size: 0.7rem; }
.finding-dir { flex-shrink: 0; font-size: 0.7rem; }
.contradiction-row {
    background: rgba(255,170,0,0.04); border: 1px solid rgba(255,170,0,0.12);
    border-radius: 2px; padding: 4px 8px; margin: 3px 0; font-size: 0.68rem; color: #664400;
}

.guideline-row { display: flex; gap: 8px; align-items: flex-start; margin: 3px 0; font-size: 0.7rem; }
.guideline-org {
    flex-shrink: 0; font-weight: 600; font-size: 0.65rem;
    font-family: 'JetBrains Mono', monospace;
}
.conflict-supports { color: #00cc88; }
.conflict-minor    { color: #ffaa00; }
.conflict-moderate { color: #ff6600; }
.conflict-major    { color: #ff4444; }

.complete-banner {
    background: rgba(0,255,170,0.04);
    border: 1px solid rgba(0,255,170,0.15);
    border-radius: 2px;
    padding: 9px 12px; margin-top: 12px;
    color: #00cc88; font-size: 0.7rem; font-weight: 500;
    display: flex; align-items: center; gap: 8px;
    font-family: 'JetBrains Mono', monospace; letter-spacing: 0.5px;
}
.complete-banner::before { content: '✓ '; }

.empty-state {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; height: 300px; color: #222; gap: 10px; text-align: center;
}
.empty-state .icon { font-size: 1.8rem; opacity: .15; }
.empty-state p { font-size: 0.72rem; margin: 0; font-family: 'JetBrains Mono', monospace; letter-spacing: 1px; }

/* ── Misc ── */
.hide-label > label, .hide-label label.svelte-1b6s6g { display: none !important; }
.synth-panel > label, .log-raw > label { display: none !important; }
.citations-accordion {
    background: #0f0f0f !important; border: 1px solid #1e1e1e !important;
    border-radius: 2px !important; margin-top: 10px !important;
}
footer { display: none !important; }

/* ── Mode radio ── */
.mode-radio .wrap { gap: 6px !important; flex-wrap: nowrap !important; }
.mode-radio .wrap label {
    border-radius: 2px !important; padding: 5px 16px !important;
    font-size: 0.72rem !important; font-weight: 500 !important;
    cursor: pointer !important; transition: all 0.15s !important;
    font-family: 'JetBrains Mono', monospace !important; letter-spacing: 0.5px !important;
}

/* ── Example chips layout ── */
.examples-row > div { flex: 1 1 0% !important; min-width: 0 !important; }
.example-btn { width: 100% !important; }
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


def _render(  # pylint: disable=too-many-arguments,R0917
    q_state: str, q_detail: str,
    r_state: str, r_detail: str,
    qa_state: str, qa_detail: str,
    an_state: str, an_detail: str,
    gl_state: str, gl_detail: str,
    sy_state: str, sy_detail: str,
    banner: str = "",
) -> str:
    parallel_block = (
        f'<div class="agent-row">'
        f'  <div class="branch-line"></div>'
        f'  <div class="parallel-group">'
        f'    <div class="parallel-label">// parallel</div>'
        f'    {_agent_card("📊", "Quality Agent", qa_state, qa_detail, qa_state == "active")}'
        f'    {_agent_card("🔬", "Analysis Agent", an_state, an_detail, an_state == "active")}'
        f'    {_agent_card("📋", "Guidelines Agent", gl_state, gl_detail, gl_state == "active")}'
        f'  </div>'
        f'</div>'
    )
    return (
        '<div class="panel-wrap">'
        '  <div class="panel-hdr">agent orchestration</div>'
        '  <div class="agent-tree">'
        '    <div class="orchestrator-header">'
        '      <div class="orch-dot"></div> ORCHESTRATOR'
        '    </div>'
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
        '  </div>'
        '</div>'
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


def _sigma_filter(
    raw: List[Paper], question: str, sigma_threshold: float = 1.2
) -> Tuple[List[Paper], str]:
    """Run σ-RAG on raw papers; return (filtered_papers, sigma_html_table)."""
    sigma_rows = ""

    def _cb(title: str, score: float, passed: bool) -> None:
        nonlocal sigma_rows
        sigma_rows += _sigma_row(title, score, passed)

    filtered = _sf.filter_papers(raw, question, sigma_threshold=sigma_threshold,
                                  max_results=12, progress_callback=_cb)
    if not filtered and raw:
        sigma_rows = ""
        fallback = max(0.3, sigma_threshold - 0.5)
        filtered = _sf.filter_papers(raw, question, sigma_threshold=fallback,
                                      max_results=12, progress_callback=_cb)

    table = (
        '<table class="sigma-table"><thead><tr>'
        '<th>Paper</th><th>σ score</th><th></th>'
        f'</tr></thead><tbody>{sigma_rows}</tbody></table>'
    )
    return filtered, table


def _run_retrieval(
    strategies: dict, question: str, sigma_threshold: float = 1.2
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
    filtered, sigma_table = _sigma_filter(raw, question, sigma_threshold)
    return raw, filtered, "  ".join(r_detail_parts), sigma_table


def _q_detail_html(strategies: dict) -> str:
    """Build the Query Agent detail block from strategy names and queries."""
    flat = [q_ for qs in strategies.values() for q_ in qs]
    tags = " ".join(_tag(n.replace("_", " "), "blue") for n in strategies)
    query_spans = "  ".join(
        f'<span class="mono" style="color:#333">{q_[:55]}…</span>'
        if len(q_) > 55 else f'<span class="mono" style="color:#333">{q_}</span>'
        for q_ in flat[:6]
    )
    return f'{tags}<br>{query_spans}'


def _verdict_badge(mode: str, verdict: str) -> str:
    """Return an HTML badge for the debunker verdict, or empty string."""
    if mode != "debunker" or not verdict:
        return ""
    color = _VERDICT_COLOR.get(verdict, "#555")
    label = _VERDICT_LABEL.get(verdict, verdict)
    return (
        f'<span style="background:#111;border:1px solid {color};'
        f'color:{color};padding:2px 10px;border-radius:2px;font-size:0.65rem;'
        f'font-weight:700;margin-top:4px;display:inline-block;'
        f'font-family:\'JetBrains Mono\',monospace;letter-spacing:1px">{label}</span>'
    )


def _citations_md(papers: List[Paper]) -> str:
    """Format filtered papers as markdown citation list with PubMed links."""
    return "\n\n".join(
        f"**[{i + 1}]** {p['authors']} ({p['year']}). "
        f"*{p['title']}*. {p['journal']}. "
        f"[PubMed {p['pmid']}]({p['url']})"
        for i, p in enumerate(papers)
    )


def _strip_citations_section(text: str) -> str:
    """Remove the ## CITATIONS block from synthesis output (shown separately in accordion)."""
    for marker in ("\n## CITATIONS", "\n## Citation", "## CITATIONS", "## Citation"):
        idx = text.find(marker)
        if idx != -1:
            return text[:idx].rstrip()
    return text


def _run_parallel_agents(
    filtered: List[Paper], question: str, mode: str, client: anthropic.Anthropic
) -> Tuple[List[StudyQuality], List[Finding], List[Contradiction], List[GuidelineConflict]]:
    """Run Quality, Analysis, and Guidelines agents concurrently."""
    quality_scores: List[StudyQuality] = []
    findings: List[Finding] = []
    contradictions: List[Contradiction] = []
    guideline_conflicts: List[GuidelineConflict] = []

    with ThreadPoolExecutor(max_workers=3) as pool:
        q_fut = pool.submit(quality.run, filtered)
        a_fut = pool.submit(analysis.run, filtered, question, mode, client)
        g_fut = pool.submit(guidelines.run, filtered, question, [], client)
        for fut in as_completed([q_fut, a_fut, g_fut]):
            result = fut.result()
            if fut is q_fut:
                quality_scores = result
            elif fut is a_fut:
                findings, contradictions = result
            else:
                guideline_conflicts = result

    return quality_scores, findings, contradictions, guideline_conflicts


def _gl_detail_html(guideline_conflicts: List[GuidelineConflict]) -> str:
    """Build the Guidelines Agent detail block."""
    if not guideline_conflicts:
        return '<span style="color:#2a2a2a;font-size:0.68rem;font-family:\'JetBrains Mono\',monospace">no conflicts detected</span>'
    level_color = {
        "supports": "conflict-supports", "minor": "conflict-minor",
        "moderate": "conflict-moderate", "major": "conflict-major",
    }
    rows = []
    for c in guideline_conflicts[:4]:
        cls = level_color.get(c["conflict_level"], "conflict-minor")
        rows.append(
            f'<div class="guideline-row">'
            f'<span class="guideline-org {cls}">[{c["organization"]}]</span>'
            f'<span style="color:#555">'
            f'{c["recommendation"][:80]}{"…" if len(c["recommendation"]) > 80 else ""}'
            f'</span>'
            f'</div>'
        )
    return "".join(rows)


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
    dir_color = {"positive": "#00cc88", "negative": "#ff4444", "neutral": "#555"}
    findings_html = "".join(
        f'<div class="finding-row">'
        f'<span class="finding-dir" style="color:{dir_color.get(f["direction"], "#555")}">'
        f'{dir_icon.get(f["direction"], "→")}</span>'
        f'<span style="color:#555">'
        f'{f["claim"][:90]}{"…" if len(f["claim"]) > 90 else ""}</span>'
        f'</div>'
        for f in findings[:5]
    )
    contra_html = "".join(
        f'<div class="contradiction-row">⚡ {c["topic"]}</div>'
        for c in contradictions[:3]
    )
    return findings_html + (contra_html if contradictions else "")


def stream_pipeline(  # pylint: disable=too-many-locals
    question: str, mode: str, sigma_threshold: float = 1.2
) -> Generator[Tuple[str, str, str], None, None]:
    """Run the multi-agent pipeline and yield (log_html, synthesis_md, citations_md)."""

    synthesis_md = ""
    citations_md = ""

    # Initial state — all pending
    def emit(  # pylint: disable=too-many-arguments,R0917
        q=("pending", ""), r=("pending", ""),
        qa=("pending", ""), an=("pending", ""),
        gl=("pending", ""), sy=("pending", ""),
        banner=""
    ) -> Tuple[str, str, str]:
        synth_out = (_SYNTH_HDR + synthesis_md) if synthesis_md else _EMPTY_SYNTH
        return (
            _render(
                q[0], q[1], r[0], r[1],
                qa[0], qa[1], an[0], an[1],
                gl[0], gl[1], sy[0], sy[1],
                banner,
            ),
            synth_out,
            citations_md,
        )

    if not question.strip():
        err = '<p style="color:#ff4444;font-size:0.75rem;font-family:\'JetBrains Mono\',monospace">⚠ Please enter a question.</p>'
        yield err, synthesis_md, citations_md
        return

    try:
        client = _get_client()
    except ValueError as exc:
        err = f'<p style="color:#ff4444;font-size:0.75rem;font-family:\'JetBrains Mono\',monospace">❌ {exc}</p>'
        yield err, "", ""
        return

    # ── Query Agent ────────────────────────────────────────────────────────────
    yield emit(q=("active", ""))

    try:
        strategies = query.run(question, mode, client)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        yield emit(q=("error", f"❌ {exc}"))
        return

    q_detail = _q_detail_html(strategies)
    yield emit(q=("done", q_detail))

    # ── Retrieval Agent ────────────────────────────────────────────────────────
    yield emit(q=("done", q_detail), r=("active", ""))

    raw_papers, filtered, r_detail_base, sigma_table = _run_retrieval(
        strategies, question, sigma_threshold
    )
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
            + '<br><span style="color:#ffaa00;font-size:0.68rem;font-family:\'JetBrains Mono\',monospace">'
            + '⚠ no papers cleared threshold</span>'
        )
        yield emit(q=("done", q_detail), r=("error", no_pass_detail),
                   qa=("pending", ""), an=("pending", ""), gl=("pending", ""))
        return

    # ── Quality ∥ Analysis ∥ Guidelines ───────────────────────────────────────
    yield emit(
        q=("done", q_detail), r=("done", r_detail_full),
        qa=("active", ""), an=("active", ""), gl=("active", ""),
    )

    quality_scores, findings, contradictions, guideline_conflicts = _run_parallel_agents(
        filtered, question, mode, client
    )
    qa_detail = _qa_detail_html(quality_scores)
    an_detail = _an_detail_html(findings, contradictions)
    gl_detail = _gl_detail_html(guideline_conflicts)

    yield emit(
        q=("done", q_detail), r=("done", r_detail_full),
        qa=("done", qa_detail), an=("done", an_detail),
        gl=("done", gl_detail), sy=("active", ""),
    )

    # ── Synthesis Agent ────────────────────────────────────────────────────────
    try:
        synth, verdict, _ = synthesis.run(
            SynthesisInput(
                papers=filtered, scores=quality_scores, findings=findings,
                contradictions=contradictions, guideline_conflicts=guideline_conflicts,
                question=question, mode=mode,
            ),
            client,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        yield emit(
            q=("done", q_detail), r=("done", r_detail_full),
            qa=("done", qa_detail), an=("done", an_detail),
            gl=("done", gl_detail), sy=("error", f"❌ {exc}"),
        )
        return

    badge = _verdict_badge(mode, verdict or "")
    sy_detail = (
        _tag(str(len(filtered)) + " papers synthesized", "green")
        + (f"  {badge}" if badge else "")
    )
    banner = '<div class="complete-banner">all agents complete — results ready</div>'
    synthesis_md = _strip_citations_section(synth)
    citations_md = _citations_md(filtered)

    yield emit(
        q=("done", q_detail), r=("done", r_detail_full),
        qa=("done", qa_detail), an=("done", an_detail),
        gl=("done", gl_detail), sy=("done", sy_detail), banner=banner,
    )
    yield emit(
        q=("done", q_detail), r=("done", r_detail_full),
        qa=("done", qa_detail), an=("done", an_detail),
        gl=("done", gl_detail), sy=("done", sy_detail), banner=banner,
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────────

_EMPTY_LOG = (
    '<div class="panel-wrap">'
    '  <div class="panel-hdr">agent orchestration</div>'
    '  <div class="empty-state">'
    '    <div class="icon">⬡</div>'
    '    <p>pipeline will stream here</p>'
    '  </div>'
    '</div>'
)

_SYNTH_HDR = '<div class="synthesis-header">evidence synthesis</div>\n\n'

_EMPTY_SYNTH = (
    _SYNTH_HDR
    + '<div class="empty-state">'
    + '<div class="icon">≡</div>'
    + '<p>synthesis will appear here</p>'
    + '</div>'
)

with gr.Blocks(css=_CSS, theme=gr.themes.Base(), title="Clinical Literature Scout") as demo:

    gr.HTML("""
    <div class="app-header">
      <div class="app-title">Clinical Literature Scout</div>
      <div class="app-subtitle">multi-agent ai · pubmed retrieval · σ-rag · quality scoring · guideline conflict detection</div>
    </div>
    """)

    # ── Input panel ───────────────────────────────────────────────────────────
    with gr.Group(elem_classes="input-card"):
        with gr.Column():

            with gr.Row():
                mode_radio = gr.Radio(
                    choices=["scout", "debunker"], value="scout",
                    label="Mode", scale=1, elem_classes="mode-radio",
                )

            gr.HTML('<div style="height:6px"></div>')

            question_box = gr.Textbox(
                label="Your Question",
                placeholder=(
                    "E.g. What is the evidence for GLP-1 receptor agonists"
                    " in reducing cardiovascular risk?"
                ),
                lines=3, min_width=600, elem_classes="question-box",
            )

            with gr.Row():
                sigma_slider = gr.Slider(
                    minimum=0.3, maximum=2.5, value=1.2, step=0.1,
                    label="σ-RAG threshold (higher = stricter relevance filter)",
                    scale=4, elem_classes="sigma-slider",
                )
                run_btn = gr.Button(
                    "RUN →", scale=1, min_width=120,
                    elem_classes="run-btn", variant="primary",
                )

    # ── Example chips ──────────────────────────────────────────────────────────
    gr.HTML(
        '<p style="color:#00ffaa44;font-size:.6rem;font-weight:600;'
        'letter-spacing:2px;text-transform:uppercase;margin:12px 0 8px;'
        'font-family:\'JetBrains Mono\',monospace">'
        '// try an example</p>'
    )
    with gr.Row(elem_classes="examples-row"):
        for ex_q, ex_m in _EXAMPLES:
            gr.Button(ex_q, elem_classes="example-btn").click(
                fn=lambda q=ex_q, m=ex_m: (q, m),
                outputs=[question_box, mode_radio],
            )

    # ── Late-injected styles (load after Gradio's Svelte CSS — wins cascade) ───
    gr.HTML("""<style>
/* Terminal textarea */
.question-box textarea {
    background: #0a0a0a !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 2px !important;
    color: #cccccc !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 0.85rem !important;
    resize: vertical !important;
    padding: 12px 14px !important;
    line-height: 1.7 !important;
    caret-color: #00ffaa !important;
}
.question-box textarea:focus {
    border-color: #00ffaa44 !important;
    box-shadow: 0 0 0 1px rgba(0,255,170,0.08) !important;
    outline: none !important;
}
.question-box textarea::placeholder { color: #2a2a2a !important; }

/* Terminal label styling */
.question-box label span,
.sigma-slider label span,
.mode-radio label span {
    color: #00ffaa66 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}

/* Mode radio — terminal toggle */
.mode-radio .wrap label {
    background: #0a0a0a !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 2px !important;
    color: #333 !important;
    padding: 5px 16px !important;
    font-size: 0.72rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 0.5px !important;
    transition: all 0.12s !important;
    cursor: pointer !important;
}
.mode-radio .wrap label:has(input:checked) {
    background: rgba(0,255,170,0.06) !important;
    border-color: #00ffaa44 !important;
    color: #00ffaa !important;
}

/* Run button — neon green terminal style */
.run-btn button {
    background: #00ffaa !important;
    border: none !important;
    border-radius: 2px !important;
    color: #0a0a0a !important;
    font-weight: 700 !important;
    font-size: 0.75rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 2px !important;
    box-shadow: 0 0 16px rgba(0,255,170,0.25) !important;
    transition: box-shadow 0.15s !important;
}
.run-btn button:hover {
    background: #00ffcc !important;
    box-shadow: 0 0 24px rgba(0,255,170,0.45) !important;
}

/* Example chips — terminal style */
.example-btn button {
    background: #0a0a0a !important;
    border: 1px solid #1a1a1a !important;
    border-radius: 2px !important;
    color: #444 !important;
    font-size: 0.68rem !important;
    font-weight: 400 !important;
    font-family: 'JetBrains Mono', monospace !important;
    line-height: 1.5 !important;
    white-space: normal !important;
    text-align: left !important;
    min-height: 56px !important;
    height: auto !important;
    padding: 10px 12px !important;
    width: 100% !important;
    transition: all 0.12s !important;
    letter-spacing: 0.2px !important;
}
.example-btn button:hover {
    background: #0f0f0f !important;
    border-color: #00ffaa33 !important;
    color: #00ffaa88 !important;
}

/* Sigma slider track */
.sigma-slider input[type=range] {
    accent-color: #00ffaa !important;
}

/* Citations accordion */
.citations-accordion > .label-wrap {
    background: #0f0f0f !important;
    border-bottom: 1px solid #1e1e1e !important;
    color: #00ffaa66 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    border-radius: 2px 2px 0 0 !important;
    padding: 10px 14px !important;
}

/* Synthesis markdown text */
.synth-inner .prose h2::before { content: '## '; color: #00ffaa44; }
.synth-inner .prose h3::before { content: '### '; color: #444; }

/* Page background */
.gradio-container, body, #root {
    background: #0a0a0a !important;
}

/* Scrollbar styling */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0a0a0a; }
::-webkit-scrollbar-thumb { background: #1e1e1e; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #00ffaa33; }
</style>""")

    # ── Output panels ──────────────────────────────────────────────────────────
    with gr.Row(equal_height=True):
        with gr.Column(scale=3):
            log_out = gr.HTML(value=_EMPTY_LOG, elem_classes="log-raw")
        with gr.Column(scale=2):
            with gr.Group(elem_classes="synth-outer"):
                synthesis_out = gr.Markdown(
                    value=_EMPTY_SYNTH,
                    elem_classes="synth-inner hide-label",
                )

    # ── Citations accordion ─────────────────────────────────────────────────────
    with gr.Accordion("📚 Citations", open=False, elem_classes="citations-accordion"):
        citations_out = gr.Markdown(value="*Citations will appear after synthesis.*")

    run_btn.click(
        fn=stream_pipeline,
        inputs=[question_box, mode_radio, sigma_slider],
        outputs=[log_out, synthesis_out, citations_out],
        show_progress="hidden",
    )

if __name__ == "__main__":
    demo.launch()
