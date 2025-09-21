"""Retrieval pipelines supporting MemoRAG and standard RAG."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from qa_bot.config import RagConfig
from qa_bot.knowledge.base import Document
from qa_bot.utils.text import chunk_text, normalize_text

try:  # pragma: no cover - heavy optional dependency
    from memorag import MemoRAG  # type: ignore
except Exception:  # pragma: no cover - MemoRAG unavailable
    MemoRAG = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore


@dataclass(slots=True)
class Answer:
    text: str
    citations: List[str]
    snippets: List[str]
    score: float
    needs_escalation: bool
    engine: str


class BasePipeline:
    """Shared interface for all retrieval pipelines."""

    name = "rag"

    def build(self, documents: Iterable[Document], *, force: bool = False) -> None:
        raise NotImplementedError

    def answer(self, question: str, *, top_k: int = 3) -> Answer:
        raise NotImplementedError


@dataclass(slots=True)
class _Chunk:
    doc_id: str
    title: str
    text: str
    url: Optional[str]


class MemoRAGPipeline(BasePipeline):
    """Wrapper around MemoRAG with an n-gram fallback."""

    name = "memorag"

    def __init__(self, config: RagConfig, cache_dir: Path) -> None:
        self.config = config
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memo = None
        self._supports_steps = False
        if MemoRAG is not None:
            try:
                self._memo = MemoRAG(
                    mem_model=config.mem_model,
                    ret_model=config.ret_model,
                    cache_dir=str(cache_dir),
                    beacon_ratio=config.beacon_ratio,
                )
                self._supports_steps = all(
                    hasattr(self._memo, attr)
                    for attr in ("memorize", "recall", "retrieve", "generate")
                )
            except Exception:
                self._memo = None
        self._fallback_state = cache_dir / "fallback_memorag.pkl"
        self._chunks: List[_Chunk] = []

    # ----------------------------- helper methods -----------------------------
    def _fallback_chunks(self, documents: Sequence[Document]) -> None:
        self._chunks.clear()
        for doc in documents:
            pieces = chunk_text(
                doc.normalised_content(),
                max_chars=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
            )
            for piece in pieces:
                self._chunks.append(
                    _Chunk(doc_id=doc.doc_id, title=doc.title, text=piece, url=doc.url)
                )
        self._save_fallback()

    def _save_fallback(self) -> None:
        data = {"chunks": self._chunks}
        try:
            self._fallback_state.write_bytes(pickle.dumps(data))
        except Exception:  # pragma: no cover - disk errors are non fatal
            pass

    def _load_fallback(self) -> bool:
        if not self._fallback_state.exists():
            return False
        try:
            data = pickle.loads(self._fallback_state.read_bytes())
        except Exception:
            return False
        self._chunks = data.get("chunks", [])
        return bool(self._chunks)

    # ----------------------------- pipeline API -----------------------------
    def build(self, documents: Iterable[Document], *, force: bool = False) -> None:
        docs = list(documents)
        if not docs:
            self._chunks = []
            return
        if self._memo is None:
            if force or not self._load_fallback():
                self._fallback_chunks(docs)
            return
        try:
            if hasattr(self._memo, "load") and not force:
                loaded = self._memo.load(str(self.cache_dir))  # type: ignore[arg-type]
                if loaded:
                    return
        except Exception:
            pass
        try:
            self._memo.memorize(  # type: ignore[attr-defined]
                [
                    {
                        "id": doc.doc_id,
                        "title": doc.title,
                        "text": doc.normalised_content(),
                        "url": doc.url,
                    }
                    for doc in docs
                ],
                force=force,
            )
        except Exception:
            # graceful fallback when MemoRAG invocation fails
            self._memo = None
            self._fallback_chunks(docs)

    def answer(self, question: str, *, top_k: int = 3) -> Answer:
        question = normalize_text(question)
        if self._memo is not None:
            try:
                if self._supports_steps:
                    clues = self._memo.recall(question)  # type: ignore[attr-defined]
                    passages = self._memo.retrieve(clues, top_k=top_k)  # type: ignore[attr-defined]
                    response = self._memo.generate(question, passages)  # type: ignore[attr-defined]
                else:
                    response = self._memo.answer(question)  # type: ignore[attr-defined]
                metadata = getattr(response, "metadata", {}) or {}
                citation_meta = metadata.get("citations", [])
                citations = []
                snippets = []
                if isinstance(citation_meta, list):
                    for item in citation_meta:
                        if isinstance(item, dict):
                            source = item.get("source") or item.get("title")
                            if source:
                                citations.append(str(source))
                            snippet = item.get("text")
                            if snippet:
                                snippets.append(str(snippet))
                score = float(metadata.get("score", 0.0))
                needs_escalation = score < self.config.retrieval_score_min or not citations
                return Answer(
                    text=str(getattr(response, "text", "")),
                    citations=citations,
                    snippets=snippets,
                    score=score,
                    needs_escalation=needs_escalation,
                    engine=self.name,
                )
            except Exception:
                # fallback path
                pass
        if not self._chunks and not self._load_fallback():
            raise RuntimeError("MemoRAGのメモリが初期化されていません。")
        best_score = 0.0
        scored: List[Tuple[float, _Chunk]] = []
        for chunk in self._chunks:
            score = self._char_ngram_score(question, chunk.text)
            if score > 0:
                scored.append((score, chunk))
                best_score = max(best_score, score)
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]
        if not top:
            return Answer(
                text="申し訳ありません。関連する情報が見つかりませんでした。",
                citations=[],
                snippets=[],
                score=0.0,
                needs_escalation=True,
                engine=self.name,
            )
        snippets = [chunk.text.strip() for _, chunk in top]
        citations = [self._format_citation(chunk) for _, chunk in top]
        base = "\n".join(snippets)
        text = (
            "以下の情報に基づき回答します。\n"
            f"{base}\n\n出典: {' / '.join(citations)}"
        )
        score = float(top[0][0]) if top else 0.0
        return Answer(
            text=text,
            citations=citations,
            snippets=snippets,
            score=score,
            needs_escalation=score < self.config.retrieval_score_min,
            engine=self.name,
        )

    @staticmethod
    def _char_ngram_score(query: str, text: str, n: int = 3) -> float:
        query = query.replace("\n", "")
        text = text.replace("\n", "")
        if len(query) < n or len(text) < n:
            return 0.0
        query_counts = {}
        for i in range(len(query) - n + 1):
            gram = query[i : i + n]
            query_counts[gram] = query_counts.get(gram, 0) + 1
        overlap = 0
        for i in range(len(text) - n + 1):
            gram = text[i : i + n]
            overlap += min(query_counts.get(gram, 0), 1)
        normaliser = max(1, sum(query_counts.values()))
        return overlap / normaliser

    @staticmethod
    def _format_citation(chunk: _Chunk) -> str:
        if chunk.url:
            return f"{chunk.title}（{chunk.url}）"
        return chunk.title


class _SentenceEncoder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = None
        if SentenceTransformer is not None:
            try:  # pragma: no cover - downloading models is environment specific
                self.model = SentenceTransformer(model_name)
            except Exception:
                self.model = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.model is not None:
            embeddings = self.model.encode(  # type: ignore[no-any-return]
                list(texts),
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return embeddings.astype("float32")
        # deterministic hash-based fallback to keep tests light-weight
        vectors = []
        for text in texts:
            vec = np.zeros(256, dtype="float32")
            if not text:
                vectors.append(vec)
                continue
            for idx, ch in enumerate(text.encode("utf-8")):
                vec[idx % 256] += (ch % 17) / 17.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vectors.append(vec)
        return np.stack(vectors, axis=0)


class StandardRAGPipeline(BasePipeline):
    """Standard retrieval-augmented generation on CPU."""

    name = "rag"

    def __init__(self, config: RagConfig) -> None:
        self.config = config
        self.encoder = _SentenceEncoder(config.ret_model)
        self._chunks: List[_Chunk] = []
        self._embeddings: Optional[np.ndarray] = None
        self._index = None

    def build(self, documents: Iterable[Document], *, force: bool = False) -> None:
        docs = list(documents)
        self._chunks = []
        for doc in docs:
            pieces = chunk_text(
                doc.normalised_content(),
                max_chars=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
            )
            for piece in pieces:
                self._chunks.append(
                    _Chunk(doc_id=doc.doc_id, title=doc.title, text=piece, url=doc.url)
                )
        if not self._chunks:
            self._embeddings = None
            self._index = None
            return
        embeddings = self.encoder.encode([c.text for c in self._chunks])
        if embeddings.ndim != 2:
            raise RuntimeError("埋め込みの生成に失敗しました。")
        self._embeddings = embeddings
        if faiss is not None:
            try:
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatIP(dimension)
                index.add(embeddings)
                self._index = index
                return
            except Exception:
                self._index = None
        self._index = None

    def answer(self, question: str, *, top_k: int = 3) -> Answer:
        if not self._chunks:
            raise RuntimeError("ドキュメントが読み込まれていません。")
        embeddings = self._embeddings
        if embeddings is None:
            raise RuntimeError("埋め込みが初期化されていません。")
        query_vec = self.encoder.encode([normalize_text(question)])[0]
        if self._index is not None:
            scores, indices = self._index.search(np.expand_dims(query_vec, axis=0), top_k)
            scores = scores[0]
            indices = indices[0]
        else:
            sims = embeddings @ query_vec
            top_k = min(top_k, len(self._chunks))
            indices = np.argsort(sims)[::-1][:top_k]
            scores = sims[indices]
        results: List[Tuple[float, _Chunk]] = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self._chunks):
                continue
            results.append((float(score), self._chunks[int(idx)]))
        results = [item for item in results if item[0] > 0]
        if not results:
            return Answer(
                text="関連する情報が見つかりませんでした。教員に確認します。",
                citations=[],
                snippets=[],
                score=0.0,
                needs_escalation=True,
                engine=self.name,
            )
        results.sort(key=lambda x: x[0], reverse=True)
        snippets = [chunk.text.strip() for _, chunk in results[:top_k]]
        citations = [self._format_citation(chunk) for _, chunk in results[:top_k]]
        score = float(results[0][0])
        answer_text = self._compose_answer(question, snippets, citations)
        needs_escalation = score < self.config.retrieval_score_min or not citations
        return Answer(
            text=answer_text,
            citations=citations,
            snippets=snippets,
            score=score,
            needs_escalation=needs_escalation,
            engine=self.name,
        )

    @staticmethod
    def _format_citation(chunk: _Chunk) -> str:
        if chunk.url:
            return f"{chunk.title}（{chunk.url}）"
        return chunk.title

    def _compose_answer(self, question: str, snippets: Sequence[str], citations: Sequence[str]) -> str:
        """Compose a concise Japanese answer without external APIs if unavailable."""

        context = "\n".join(snippets)
        citation_text = " / ".join(citations) if citations else "情報源なし"
        if self.config.gen_provider and self.config.gen_model and self.config.openai_api_key:
            generated = self._call_openai(question, context, citation_text)
            if generated:
                return generated
        return (
            "以下の資料に基づき回答します。\n"
            f"{context}\n\n出典: {citation_text}"
        )

    def _call_openai(self, question: str, context: str, citation_text: str) -> Optional[str]:
        if self.config.gen_provider.lower() != "openai":
            return None
        try:  # pragma: no cover - network interaction
            from openai import OpenAI

            client = OpenAI(api_key=self.config.openai_api_key)
            prompt = (
                "あなたは大学講義の日本語TAボットです。"
                "次のコンテキストを引用しながら学生の質問に回答してください。"
            )
            messages = [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": f"質問: {question}\nコンテキスト:\n{context}\n出典: {citation_text}",
                },
            ]
            response = client.chat.completions.create(  # type: ignore[attr-defined]
                model=self.config.gen_model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            choice = response.choices[0]
            content = getattr(choice, "message", {}).get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
        except Exception:
            return None
        return None

