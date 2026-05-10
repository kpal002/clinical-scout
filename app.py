#!/usr/bin/env python3
"""Hugging Face Spaces demo — Clinical Literature Scout + Medical Myth Debunker."""
from __future__ import annotations

import json
import os
import sys
from typing import Generator, List, Optional

import anthropic
import gradio as gr

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from retrieval.pubmed import fetch_abstracts, search_pubmed  # noqa: E402
from retrieval.sigma_filter import filter_papers  # noqa: E402
from agent.state import Paper  # noqa: E402

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

_VERDICT_COLOR = {"BUSTED": "#22c55e", "SUPPORTED": "#ef4444", "MIXED": "#f59e0b"}
_VERDICT_LABEL = {"BUSTED": "BUSTED ✓", "SUPPORTED": "SUPPORTED", "MIXED": "MIXED EVIDENCE"}

_EXAMPLES = [
    ["What is the evidence for GLP-1 receptor agonists in reducing cardiovascular risk?", "scout"],
    ["Does metformin reduce cancer risk in diabetic patients?", "scout"],
    ["Does cracking your knuckles cause arthritis?", "debunker"],
    ["Do we only use 10% of our brain?", "debunker"],
    ["Does vitamin C prevent the common cold?", "debunker"],
]

# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

* { box-sizing: border-box; }

body, .gradio-container {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    background: #0a0a0f !important;
}

.gradio-container { max-width: 1400px !important; margin: 0 auto !important; }

/* Header */
.app-header {
    padding: 48px 0 32px;
    text-align: center;
}
.app-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 10px;
    letter-spacing: -0.5px;
}
.app-header p {
    color: #6b7280;
    font-size: 0.95rem;
    font-weight: 400;
    margin: 0;
}

/* Input row */
.input-card {
    background: #111118;
    border: 1px solid #1e1e2e;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
}

/* Mode radio */
.mode-radio label { color: #9ca3af !important; font-size: 0.8rem !important; }
.mode-radio .wrap { gap: 8px !important; }
.mode-radio .wrap label {
    background: #1a1a2e !important;
    border: 1px solid #2a2a3e !important;
    border-radius: 8px !important;
    padding: 6px 16px !important;
    color: #9ca3af !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
}
.mode-radio .wrap label:has(input:checked) {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    border-color: transparent !important;
    color: white !important;
}

/* Question box */
.question-box textarea {
    background: #0d0d1a !important;
    border: 1px solid #2a2a3e !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    resize: none !important;
}
.question-box textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important;
}

/* Run button */
.run-btn {
    background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    height: 52px !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 24px rgba(99,102,241,0.3) !important;
}
.run-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 32px rgba(99,102,241,0.45) !important;
}

/* Panel labels */
.panel-label {
    color: #4b5563;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 12px;
    padding-bottom: 10px;
    border-bottom: 1px solid #1e1e2e;
}

/* Agent log panel */
.log-panel {
    background: #0d0d18;
    border: 1px solid #1a1a2a;
    border-radius: 16px;
    padding: 24px;
    min-height: 520px;
}

/* Synthesis panel */
.synth-panel {
    background: #0d0d18;
    border: 1px solid #1a1a2a;
    border-radius: 16px;
    padding: 24px;
    min-height: 520px;
}
.synth-panel h2, .synth-panel h3 {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}
.synth-panel p, .synth-panel li { color: #94a3b8 !important; line-height: 1.7 !important; }

/* Citations accordion */
.citations-accordion {
    background: #0d0d18 !important;
    border: 1px solid #1a1a2a !important;
    border-radius: 12px !important;
    margin-top: 12px;
}

/* Example buttons */
.example-btn button {
    background: #111118 !important;
    border: 1px solid #2a2a3e !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important;
    transition: all 0.2s !important;
    white-space: normal !important;
    text-align: left !important;
    height: auto !important;
    padding: 8px 12px !important;
    line-height: 1.4 !important;
}
.example-btn button:hover {
    background: #1a1a2e !important;
    border-color: #3b82f6 !important;
    color: #60a5fa !important;
}

/* Pipeline steps (rendered as HTML inside gr.HTML) */
.pipeline { display: flex; flex-direction: column; gap: 0; }

.step-item {
    display: flex;
    gap: 16px;
    position: relative;
}
.step-item:not(:last-child)::before {
    content: '';
    position: absolute;
    left: 15px;
    top: 36px;
    bottom: -8px;
    width: 2px;
    background: linear-gradient(to bottom, #2a2a3e, transparent);
}

.step-dot {
    flex-shrink: 0;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    margin-top: 4px;
    position: relative;
    z-index: 1;
}
.step-dot.done {
    background: linear-gradient(135deg, #059669, #34d399);
    color: white;
    box-shadow: 0 0 12px rgba(52,211,153,0.3);
}
.step-dot.active {
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    color: white;
    box-shadow: 0 0 16px rgba(99,102,241,0.5);
    animation: pulse-dot 1.5s ease-in-out infinite;
}
.step-dot.pending {
    background: #1a1a2a;
    border: 1px solid #2a2a3e;
    color: #4b5563;
}

@keyframes pulse-dot {
    0%, 100% { box-shadow: 0 0 16px rgba(99,102,241,0.5); }
    50% { box-shadow: 0 0 28px rgba(99,102,241,0.8); }
}

.step-body { flex: 1; padding-bottom: 24px; }

.step-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
    line-height: 1.4;
    padding-top: 6px;
}
.step-title.active { color: #60a5fa; }
.step-title.done { color: #34d399; }

.thinking-dots {
    display: inline-flex;
    gap: 3px;
    align-items: center;
}
.thinking-dots span {
    width: 4px; height: 4px;
    background: #6366f1;
    border-radius: 50%;
    animation: bounce 1.2s ease-in-out infinite;
}
.thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
.thinking-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
    0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
    40% { transform: translateY(-4px); opacity: 1; }
}

.step-content { margin-top: 6px; }

.query-tag {
    display: inline-block;
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 6px;
    padding: 3px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #7dd3fc;
    margin: 2px 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 100%;
}

.result-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 4px 0;
    font-size: 0.82rem;
    color: #6b7280;
}
.result-count {
    color: #60a5fa;
    font-weight: 600;
    font-size: 0.82rem;
}

.sigma-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 8px;
    font-size: 0.78rem;
}
.sigma-table th {
    color: #4b5563;
    font-weight: 500;
    text-align: left;
    padding: 4px 8px;
    border-bottom: 1px solid #1e1e2e;
    font-size: 0.7rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.sigma-table td {
    padding: 5px 8px;
    color: #6b7280;
    border-bottom: 1px solid #111118;
    vertical-align: middle;
}
.sigma-table td.pass { color: #34d399; font-weight: 500; }
.sigma-table td.fail { color: #4b5563; }
.sigma-score-pass { color: #34d399; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; }
.sigma-score-fail { color: #374151; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; }

.stat-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #111118;
    border: 1px solid #1e1e2e;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.78rem;
    color: #6b7280;
    margin: 2px;
}
.stat-pill strong { color: #e2e8f0; }

.verdict-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.5px;
    margin-top: 8px;
}

.complete-banner {
    background: linear-gradient(135deg, rgba(5,150,105,0.1), rgba(52,211,153,0.05));
    border: 1px solid rgba(52,211,153,0.2);
    border-radius: 10px;
    padding: 12px 16px;
    margin-top: 16px;
    color: #34d399;
    font-size: 0.82rem;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 8px;
}

.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 300px;
    color: #374151;
    gap: 12px;
    text-align: center;
}
.empty-state .icon { font-size: 2.5rem; opacity: 0.3; }
.empty-state p { font-size: 0.85rem; margin: 0; }

/* Hide gradio label clutter */
.hide-label > label { display: none !important; }
footer { display: none !important; }
"""

# ── HTML helpers ──────────────────────────────────────────────────────────────

def _dot(state: str, num: int) -> str:
    icons = {"done": "✓", "active": str(num), "pending": str(num)}
    return f'<div class="step-dot {state}">{icons[state]}</div>'


def _thinking() -> str:
    return (
        '<span class="thinking-dots">'
        '<span></span><span></span><span></span>'
        '</span>'
    )


def _step_html(
    num: int,
    icon: str,
    title: str,
    state: str,
    body: str = "",
    last: bool = False,
) -> str:
    title_class = {"done": "done", "active": "active", "pending": ""}.get(state, "")
    suffix = _thinking() if state == "active" else ""
    connector = "" if last else ""
    return f"""
<div class="step-item" {'style="padding-bottom:0"' if last else ''}>
  {_dot(state, num)}
  <div class="step-body" {'style="padding-bottom:8px"' if last else ''}>
    <div class="step-title {title_class}">{icon} {title} {suffix}</div>
    <div class="step-content">{body}</div>
  </div>
</div>
{connector}"""


def _pipeline_html(steps: list[dict]) -> str:
    items = ""
    for i, s in enumerate(steps):
        items += _step_html(
            num=i + 1,
            icon=s["icon"],
            title=s["title"],
            state=s["state"],
            body=s.get("body", ""),
            last=(i == len(steps) - 1),
        )
    return f'<div class="pipeline">{items}</div>'


# ── Pipeline ──────────────────────────────────────────────────────────────────

def _client() -> anthropic.Anthropic:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY is not set in Space secrets.")
    return anthropic.Anthropic(api_key=key)


def _formulate_queries(question: str, mode: str) -> List[str]:
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
    label = "Medical Myth" if mode == "debunker" else "Clinical Question"
    numbered = "\n\n".join(
        f"[{i + 1}] {p['authors']} ({p['year']}) — {p['journal']}\n"
        f"Title: {p['title']}\nAbstract: {p['abstract'][:1200]}"
        for i, p in enumerate(papers)
    )
    resp = _client().messages.create(
        model=_MODEL,
        max_tokens=2048,
        system=_SYNTHESIS_SYSTEM[mode],
        messages=[{"role": "user", "content": f"{label}: {question}\n\nPapers:\n{numbered}"}],
    )
    return resp.content[0].text.strip()


def _extract_verdict(synthesis: str) -> Optional[str]:
    for line in synthesis.splitlines():
        upper = line.upper()
        if "BUSTED" in upper:
            return "BUSTED"
        if "SUPPORTED" in upper:
            return "SUPPORTED"
        if "MIXED" in upper:
            return "MIXED"
    return None


def stream_pipeline(
    question: str, mode: str
) -> Generator[tuple[str, str, str], None, None]:

    synthesis = ""
    citations = ""

    def _render(steps: list[dict], banner: str = "") -> str:
        return _pipeline_html(steps) + banner

    if not question.strip():
        err = '<p style="color:#ef4444;font-size:0.85rem">⚠ Please enter a question.</p>'
        yield err, synthesis, citations
        return

    try:
        _client()
    except ValueError as exc:
        err = f'<p style="color:#ef4444;font-size:0.85rem">❌ {exc}</p>'
        yield err, "", ""
        return

    steps = [
        {"icon": "🔍", "title": "Formulating PubMed Queries", "state": "active", "body": ""},
        {"icon": "📡", "title": "Searching PubMed", "state": "pending", "body": ""},
        {"icon": "📄", "title": "Fetching Abstracts", "state": "pending", "body": ""},
        {"icon": "🔬", "title": "σ-RAG Significance Filter", "state": "pending", "body": ""},
        {"icon": "🧠", "title": "Evidence Synthesis", "state": "pending", "body": ""},
    ]
    yield _render(steps), synthesis, citations

    # Step 1
    try:
        queries = _formulate_queries(question, mode)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        steps[0]["body"] = f'<p style="color:#ef4444;font-size:0.8rem">❌ {exc}</p>'
        steps[0]["state"] = "done"
        yield _render(steps), synthesis, citations
        return

    q_html = "".join(
        f'<div style="margin:4px 0"><span class="query-tag">{q}</span></div>'
        for q in queries
    )
    steps[0]["body"] = q_html
    steps[0]["state"] = "done"
    steps[1]["state"] = "active"
    yield _render(steps), synthesis, citations

    # Step 2
    all_pmids: set[str] = set()
    result_rows = ""
    for q in queries:
        try:
            pmids = search_pubmed(q, max_results=_MAX_PER_QUERY)
        except Exception:  # pylint: disable=broad-exception-caught
            pmids = []
        all_pmids.update(pmids)
        short_q = (q[:55] + "…") if len(q) > 55 else q
        result_rows += (
            f'<div class="result-row">'
            f'<span class="query-tag" style="max-width:320px">{short_q}</span>'
            f'<span>→</span>'
            f'<span class="result-count">{len(pmids)}</span>'
            f'<span style="color:#4b5563">results</span>'
            f'</div>'
        )
        steps[1]["body"] = result_rows
        yield _render(steps), synthesis, citations

    unique_pmids = list(all_pmids)
    steps[1]["body"] = result_rows + (
        f'<div class="stat-pill" style="margin-top:8px">'
        f'Total unique PMIDs: <strong>{len(unique_pmids)}</strong>'
        f'</div>'
    )
    steps[1]["state"] = "done"
    steps[2]["state"] = "active"
    yield _render(steps), synthesis, citations

    if not unique_pmids:
        steps[2]["body"] = '<p style="color:#f59e0b;font-size:0.8rem">⚠ No results. Try rephrasing.</p>'
        steps[2]["state"] = "done"
        yield _render(steps), synthesis, citations
        return

    # Step 3
    try:
        papers = fetch_abstracts(unique_pmids)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        steps[2]["body"] = f'<p style="color:#ef4444;font-size:0.8rem">❌ {exc}</p>'
        steps[2]["state"] = "done"
        yield _render(steps), synthesis, citations
        return

    no_abstract = len(unique_pmids) - len(papers)
    steps[2]["body"] = (
        f'<div class="stat-pill"><strong>{len(papers)}</strong> abstracts fetched</div>'
        f'<div class="stat-pill" style="color:#4b5563"><strong>{no_abstract}</strong> had none</div>'
    )
    steps[2]["state"] = "done"
    steps[3]["state"] = "active"
    yield _render(steps), synthesis, citations

    # Step 4
    threshold = _SIGMA_THRESHOLD
    table_rows = ""

    def _cb(title: str, score: float, passed: bool) -> None:
        nonlocal table_rows
        short = (title[:60] + "…") if len(title) > 60 else title
        icon = "✓" if passed else "✗"
        score_cls = "sigma-score-pass" if passed else "sigma-score-fail"
        td_cls = "pass" if passed else "fail"
        table_rows += (
            f"<tr>"
            f'<td class="{td_cls}" style="max-width:340px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{short}</td>'
            f'<td><span class="{score_cls}">{score:.2f}σ</span></td>'
            f'<td class="{td_cls}">{icon}</td>'
            f"</tr>"
        )

    def _table() -> str:
        return (
            '<table class="sigma-table">'
            '<thead><tr><th>Paper</th><th>Score</th><th></th></tr></thead>'
            f'<tbody>{table_rows}</tbody>'
            '</table>'
        )

    steps[3]["body"] = _table()
    filtered = filter_papers(
        papers, question,
        sigma_threshold=threshold,
        max_results=_MAX_FILTERED,
        progress_callback=_cb,
    )

    if not filtered and papers:
        table_rows = ""
        threshold = _SIGMA_RETRY
        steps[3]["body"] = (
            f'<p style="color:#f59e0b;font-size:0.78rem;margin:4px 0">'
            f'Nothing passed {_SIGMA_THRESHOLD}σ — retrying at {threshold}σ…</p>'
            + _table()
        )
        yield _render(steps), synthesis, citations
        filtered = filter_papers(
            papers, question,
            sigma_threshold=threshold,
            max_results=_MAX_FILTERED,
            progress_callback=_cb,
        )

    steps[3]["body"] = (
        _table()
        + f'<div style="margin-top:10px;display:flex;flex-wrap:wrap;gap:4px">'
        f'<div class="stat-pill"><strong>{len(papers)}</strong> retrieved</div>'
        f'<div class="stat-pill" style="color:#34d399"><strong>{len(filtered)}</strong> passed</div>'
        f'<div class="stat-pill" style="color:#4b5563"><strong>{len(papers) - len(filtered)}</strong> filtered</div>'
        f'</div>'
    )
    steps[3]["state"] = "done"
    steps[4]["state"] = "active"
    yield _render(steps), synthesis, citations

    if not filtered:
        steps[4]["body"] = '<p style="color:#f59e0b;font-size:0.8rem">⚠ No papers cleared threshold. Try rephrasing.</p>'
        steps[4]["state"] = "done"
        yield _render(steps), synthesis, citations
        return

    # Step 5
    try:
        synthesis = _synthesize(filtered, question, mode)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        steps[4]["body"] = f'<p style="color:#ef4444;font-size:0.8rem">❌ {exc}</p>'
        steps[4]["state"] = "done"
        yield _render(steps), synthesis, citations
        return

    verdict_html = ""
    if mode == "debunker":
        verdict = _extract_verdict(synthesis)
        if verdict:
            color = _VERDICT_COLOR.get(verdict, "#6b7280")
            label = _VERDICT_LABEL.get(verdict, verdict)
            verdict_html = (
                f'<div class="verdict-badge" style="background:rgba(0,0,0,0.4);'
                f'border:1px solid {color};color:{color}">{label}</div>'
            )

    steps[4]["body"] = (
        f'<div class="stat-pill"><strong>{len(filtered)}</strong> high-signal abstracts synthesized</div>'
        + verdict_html
    )
    steps[4]["state"] = "done"

    banner = (
        '<div class="complete-banner">'
        '✦ Pipeline complete — results ready'
        '</div>'
    )
    yield _render(steps, banner), synthesis, citations

    citations = "\n\n".join(
        f"**[{i + 1}]** {p['authors']} ({p['year']}). "
        f"*{p['title']}*. {p['journal']}. "
        f"[PubMed {p['pmid']}]({p['url']})"
        for i, p in enumerate(filtered)
    )
    yield _render(steps, banner), synthesis, citations


# ── UI ────────────────────────────────────────────────────────────────────────

_EMPTY_LOG = """
<div class="empty-state">
  <div class="icon">🔬</div>
  <p>Agent reasoning will stream here step by step</p>
</div>
"""

_EMPTY_SYNTH = """
<div class="empty-state">
  <div class="icon">📋</div>
  <p>Evidence synthesis will appear here</p>
</div>
"""

with gr.Blocks(css=_CSS, title="Clinical Literature Scout") as demo:

    gr.HTML("""
    <div class="app-header">
      <h1>Clinical Literature Scout</h1>
      <p>Multi-step AI agent &nbsp;·&nbsp; σ-RAG + PubMed + Claude</p>
    </div>
    """)

    with gr.Group(elem_classes="input-card"):
        with gr.Row():
            mode_radio = gr.Radio(
                choices=["scout", "debunker"],
                value="scout",
                label="Mode",
                info="Scout = clinical evidence  ·  Debunker = myth verification",
                scale=1,
                elem_classes="mode-radio",
            )
            question_box = gr.Textbox(
                label="Your question or myth",
                placeholder="E.g. What is the evidence for GLP-1 receptor agonists in reducing cardiovascular risk?",
                lines=2,
                scale=4,
                elem_classes="question-box",
            )

    run_btn = gr.Button("Run Agent →", variant="primary", size="lg", elem_classes="run-btn")

    with gr.Row(equal_height=True):
        with gr.Column(scale=3):
            gr.HTML('<div class="panel-label">🤖 &nbsp;Agent Reasoning</div>')
            log_out = gr.HTML(value=_EMPTY_LOG, elem_classes="log-panel hide-label")
        with gr.Column(scale=2):
            gr.HTML('<div class="panel-label">📋 &nbsp;Evidence Synthesis</div>')
            synthesis_out = gr.Markdown(
                value=_EMPTY_SYNTH,
                elem_classes="synth-panel hide-label",
            )

    with gr.Accordion("📚 Citations", open=False, elem_classes="citations-accordion"):
        citations_out = gr.Markdown(value="*Citations will appear after synthesis.*")

    gr.HTML('<div class="panel-label" style="margin-top:24px">💡 &nbsp;Try an example</div>')
    with gr.Row():
        for ex_q, ex_m in _EXAMPLES:
            short = ex_q[:60] + "…" if len(ex_q) > 60 else ex_q
            gr.Button(short, size="sm", elem_classes="example-btn").click(
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
