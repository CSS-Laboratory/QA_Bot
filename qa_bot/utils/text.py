"""Utilities for normalising and chunking Japanese text."""
from __future__ import annotations

import unicodedata
from typing import Iterable, List


DEFAULT_SEPARATORS = ["\n\n", "。", "！", "？", "\n"]


def normalize_text(text: str) -> str:
    """Return the text normalised using NFKC."""
    return unicodedata.normalize("NFKC", text)


def chunk_text(
    text: str,
    *,
    max_chars: int = 900,
    overlap: int = 120,
    separators: Iterable[str] = DEFAULT_SEPARATORS,
) -> List[str]:
    """Split *text* into overlapping chunks suited for retrieval."""
    if not text:
        return []

    separators = list(separators)
    chunks: List[str] = []
    remaining = text
    while remaining:
        chunk = remaining[:max_chars]
        for sep in separators:
            idx = chunk.rfind(sep)
            if idx != -1 and idx >= max_chars // 3:
                chunk = chunk[: idx + len(sep)]
                break
        chunks.append(chunk.strip())
        if len(remaining) <= max_chars:
            break
        start = max(0, len(chunk) - overlap)
        remaining = remaining[start:]
    return [c for c in chunks if c]
