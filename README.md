---
title: Clinical Literature Scout
emoji: 🔬
colorFrom: blue
colorTo: cyan
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: mit
short_description: Multi-step AI agent — σ-RAG + PubMed + Claude
---

# 🔬 Clinical Literature Scout + Medical Myth Debunker

A multi-step AI agent that searches PubMed, applies **σ-RAG significance filtering**, and synthesizes evidence using Claude.

## How it works

| Step | Tool | What happens |
|------|------|-------------|
| 1 | **Claude** | Translates your question into 3 optimised PubMed MeSH queries |
| 2 | **NCBI Esearch** | Searches PubMed and collects unique paper IDs |
| 3 | **NCBI Efetch** | Downloads full abstracts for each paper |
| 4 | **σ-RAG** | Fits a Gaussian noise floor; keeps only papers above the significance threshold |
| 5 | **Claude** | Synthesizes a structured evidence report with citations |

σ-RAG eliminates noise by treating retrieval as a signal-detection problem — only papers whose relevance score is statistically above background are passed to the LLM, preventing hallucination from irrelevant context.

## Modes

- **Scout** — for clinical questions (evidence synthesis with citations)
- **Debunker** — for medical myths (verdict: BUSTED / SUPPORTED / MIXED)

## Setup

Add your `ANTHROPIC_API_KEY` in **Settings → Secrets**.
