"""Knowledge source abstractions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from qa_bot.utils.text import normalize_text


@dataclass(slots=True)
class Document:
    doc_id: str
    title: str
    content: str
    url: str | None = None

    def normalised_content(self) -> str:
        return normalize_text(self.content)


class KnowledgeSource(Protocol):
    """Protocol describing an object able to yield documents."""

    def load_documents(self) -> Iterable[Document]:
        ...
