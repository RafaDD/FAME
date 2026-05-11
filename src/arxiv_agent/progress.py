"""Shared progress logging for arxiv_agent."""
from __future__ import annotations


def _progress(message: str) -> None:
    print(f"[arxiv_agent] {message}", flush=True)
