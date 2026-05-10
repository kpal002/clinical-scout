"""Gradio CSS — terminal theme for Clinical Literature Scout."""

CSS = """
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
.status-dot.active  {
    background: #00ffaa; box-shadow: 0 0 6px #00ffaa88;
    animation: blink 1s ease-in-out infinite;
}
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
@keyframes bounce {
    0%,80%,100%{transform:translateY(0);opacity:.3}
    40%{transform:translateY(-4px);opacity:1}
}

.agent-detail {
    font-size: 0.68rem; color: #444; line-height: 1.6;
    padding-left: 14px; font-family: 'JetBrains Mono', monospace;
}
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

.finding-row {
    display: flex; gap: 6px; align-items: flex-start; margin: 3px 0; font-size: 0.7rem;
}
.finding-dir { flex-shrink: 0; font-size: 0.7rem; }
.contradiction-row {
    background: rgba(255,170,0,0.04); border: 1px solid rgba(255,170,0,0.12);
    border-radius: 2px; padding: 4px 8px; margin: 3px 0;
    font-size: 0.68rem; color: #664400;
}

.guideline-row {
    display: flex; gap: 8px; align-items: flex-start; margin: 3px 0; font-size: 0.7rem;
}
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
.empty-state p {
    font-size: 0.72rem; margin: 0;
    font-family: 'JetBrains Mono', monospace; letter-spacing: 1px;
}

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
#examples-row {
    display: grid !important;
    grid-template-columns: repeat(5, 1fr) !important;
    gap: 8px !important;
    align-items: stretch !important;
}
#examples-row > * {
    height: 100px !important;
}
"""
