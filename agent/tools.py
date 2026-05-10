"""Shared client factory + backward-compat re-exports for the multi-agent pipeline."""
from __future__ import annotations

import functools
import os

import anthropic
from rich.console import Console

console = Console()


@functools.lru_cache(maxsize=1)
def _client() -> anthropic.Anthropic:
    """Return a cached Anthropic client (one instance per process)."""
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def reset_client_cache() -> None:
    """Evict the cached Anthropic client (useful in tests to inject mocks)."""
    _client.cache_clear()
