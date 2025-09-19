"""MemoRAG wrapper with a lightweight local fallback implementation."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from qa_bot.config import RagConfig
from qa_bot.knowledge.base import Document
from qa_bot.utils.text import chunk_text, normalize_text

try:  # pragma: no cover - optional dependency
    from memorag import MemoRAG  # type: ignore
except Exception:  # pragma: no cover
    MemoRAG = None


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    url: str | None
    ngrams: Dict[str, int]


@dataclass(slots=True)
class Answer:
    text: str
    citations: List[str]
    snippets: List[str]
    score: float
    needs_escalation: bool


class MemoRAGClient:
    """Facade hiding the difference between MemoRAG and the fallback."""

    def __init__(self, config: RagConfig, cache_dir: Path) -> None:
        self.config = config
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.cache_dir / "fallback_memorag.pkl"
        self._memo_rag = None
        if MemoRAG is not None:
            self._memo_rag = MemoRAG(
                mem_model=config.mem_model,
                ret_model=config.ret_model,
                gen_model=config.gen_model or "gpt-5-nano",
                cache_dir=str(cache_dir),
            )
        self._chunks: List[Chunk] = []
        self._doc_titles: Dict[str, str] = {}

    # ------------------------- public API -------------------------
    def build_memory(self, documents: Iterable[Document], *, force: bool = False) -> None:
        docs = list(documents)
        if self._memo_rag is not None:
            self._memo_rag.memorize(docs, force=force)
            return
        if not force and self.load_state():
            return
        self._chunks = []
        self._doc_titles = {doc.doc_id: doc.title for doc in docs}
        for doc in docs:
            for index, chunk in enumerate(chunk_text(doc.normalised_content())):
                chunk_id = f"{doc.doc_id}-{index}"
                self._chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc.doc_id,
                        title=doc.title,
                        text=chunk,
                        url=doc.url,
                        ngrams=self._to_ngrams(chunk),
                    )
                )
        self.save_state()

    def answer(self, question: str, top_k: int = 3) -> Answer:
        question = normalize_text(question)
        if self._memo_rag is not None:
            response = self._memo_rag.answer(question)
            citation_meta = response.metadata.get('citations', [])
            citations = [c.get('source') for c in citation_meta if isinstance(c, dict)]
            snippets = [c.get('text', '') for c in citation_meta if isinstance(c, dict) and c.get('text')]
            score = float(response.metadata.get('score', 0.0))
            needs_escalation = score < self.config.retrieval_score_min or not citations
            return Answer(
                text=response.text,
                citations=[c for c in citations if c],
                snippets=snippets,
                score=score,
                needs_escalation=needs_escalation,
            )
        if not self._chunks and not self.load_state():
            raise RuntimeError("メモリが初期化されていません。build_memory() を先に呼んでください。")
        scored = [
            (self._score(question, chunk.ngrams), chunk)
            for chunk in self._chunks
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [item for item in scored[:top_k] if item[0] > 0]
        if not top:
            return Answer(
                text="申し訳ありません。この質問に関する十分な情報が見つかりませんでした。",
                citations=[],
                snippets=[],
                score=0.0,
                needs_escalation=True,
            )
        snippets = [chunk.text.strip() for _, chunk in top]
        summary = "\n".join(snippets)
        citations = [self._format_citation(chunk) for _, chunk in top]
        score = float(top[0][0])
        needs_escalation = score < self.config.retrieval_score_min
        answer_text = self._compose_answer(summary, citations)
        return Answer(
            text=answer_text,
            citations=citations,
            snippets=snippets,
            score=score,
            needs_escalation=needs_escalation,
        )

    def load_state(self) -> bool:
        if not self.state_path.exists():
            return False
        data = pickle.loads(self.state_path.read_bytes())
        self._chunks = data["chunks"]
        self._doc_titles = data["doc_titles"]
        return True

    def save_state(self) -> None:
        data = {"chunks": self._chunks, "doc_titles": self._doc_titles}
        self.state_path.write_bytes(pickle.dumps(data))

    # ---------------------- fallback helpers ----------------------
    @staticmethod
    def _to_ngrams(text: str, size: int = 3) -> Dict[str, int]:
        text = text.replace("\n", "")
        if len(text) < size:
            return {text: 1}
        grams: Dict[str, int] = {}
        for i in range(len(text) - size + 1):
            gram = text[i : i + size]
            grams[gram] = grams.get(gram, 0) + 1
        return grams

    @staticmethod
    def _score(query: str, chunk_ngrams: Dict[str, int], size: int = 3) -> float:
        query_ngrams = MemoRAGClient._to_ngrams(query, size=size)
        overlap = 0
        total = sum(query_ngrams.values())
        if total == 0:
            return 0.0
        for gram, count in query_ngrams.items():
            if gram in chunk_ngrams:
                overlap += min(count, chunk_ngrams[gram])
        return overlap / total

    @staticmethod
    def _format_citation(chunk: Chunk) -> str:
        if chunk.url:
            return f"{chunk.title}（{chunk.url}）"
        return chunk.title

    @staticmethod
    def _compose_answer(summary: str, citations: List[str]) -> str:
        citation_text = " / ".join(citations)
        return f"以下の情報に基づき回答します。\n{summary}\n\n出典: {citation_text}"
