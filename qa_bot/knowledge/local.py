"""Local knowledge source used in test mode."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List

from qa_bot.knowledge.base import Document
from qa_bot.utils.text import normalize_text


class LocalMarkdownKnowledgeSource:
    """Knowledge source that reads Markdown text from disk or memory."""

    def __init__(
        self,
        *,
        path: Path | None = None,
        inline_text: str | None = None,
        title: str = "ローカル知識ベース",
        base_url: str | None = None,
    ) -> None:
        if not path and not inline_text:
            raise ValueError("path または inline_text のいずれかを指定してください。")
        self.path = path
        self.inline_text = inline_text
        self.title = title
        self.base_url = base_url

    def load_documents(self) -> Iterable[Document]:
        text = self.inline_text
        if self.path:
            text = Path(self.path).read_text(encoding="utf-8")
        if text is None:
            return []
        text = normalize_text(text)
        sections = self._split_sections(text)
        documents: List[Document] = []
        for index, (heading, body) in enumerate(sections):
            slug = hashlib.sha1(heading.encode("utf-8")).hexdigest()[:10]
            doc_id = f"local-{slug}-{index}"
            url = f"{self.base_url}#{slug}" if self.base_url else None
            documents.append(
                Document(
                    doc_id=doc_id,
                    title=heading,
                    content=body.strip(),
                    url=url,
                )
            )
        if not documents:
            documents.append(
                Document(
                    doc_id="local-root",
                    title=self.title,
                    content=text,
                    url=self.base_url,
                )
            )
        return documents

    @staticmethod
    def _split_sections(text: str) -> List[tuple[str, str]]:
        sections: List[tuple[str, str]] = []
        current_heading = "全体"
        current_lines: List[str] = []
        for line in text.splitlines():
            if line.startswith("#"):
                if current_lines:
                    sections.append((current_heading, "\n".join(current_lines)))
                current_heading = line.lstrip("# ").strip() or "セクション"
                current_lines = []
            else:
                current_lines.append(line)
        if current_lines:
            sections.append((current_heading, "\n".join(current_lines)))
        return sections

