"""Core orchestration service tying together knowledge, RAG and storage."""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional

from qa_bot.config import AppConfig
from qa_bot.knowledge.base import Document
from qa_bot.knowledge.google_docs import GoogleDriveKnowledgeSource
from qa_bot.knowledge.local import LocalMarkdownKnowledgeSource
from qa_bot.rag.memorag_client import Answer, MemoRAGClient
from qa_bot.storage import (
    EmbeddingRecord,
    EmbeddingStore,
    EscalationEntry,
    EscalationLogger,
    PickleState,
    QuestionLogEntry,
    QuestionLogger,
)
from qa_bot.utils.text import normalize_text


@dataclass(slots=True)
class EscalationPayload:
    question: str
    user_id: str
    channel_id: str
    score: float
    citations: List[str]
    snippets: List[str]
    reason: str


class QABotService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.config.ensure_directories()
        self.question_logger = QuestionLogger(config.dashboard.logs_file)
        self.escalation_logger = EscalationLogger(config.dashboard.escalations_file)
        self.state = PickleState(config.dashboard.data_root / "pickle" / "state.pkl")
        self.embedding_store = EmbeddingStore(config.dashboard.embeddings_file)
        self.memo_client = MemoRAGClient(config.rag, config.knowledge.cache_dir)
        stored_faq = self.state.get("faq_master_doc_id")
        if stored_faq and not self.config.knowledge.faq_master_doc_id:
            self.config.knowledge.faq_master_doc_id = stored_faq
        stored_folder = self.state.get("google_drive_folder_id")
        if stored_folder and not self.config.knowledge.google_drive_folder_id:
            self.config.knowledge.google_drive_folder_id = stored_folder
        self.documents: List[Document] = []
        self.document_map: Dict[str, Document] = {}
        self._escalation_callback: Optional[Callable[[EscalationPayload], None]] = None

    # ------------------------------------------------------------------
    def set_escalation_callback(
        self, callback: Callable[[EscalationPayload], None] | None
    ) -> None:
        self._escalation_callback = callback

    # ------------------------------------------------------------------
    def bootstrap(self, *, force_reindex: bool = False) -> None:
        self.documents = list(self._load_documents())
        self.document_map = {doc.doc_id: doc for doc in self.documents}
        self.memo_client.build_memory(self.documents, force=force_reindex)

    def rebuild_memory(self) -> None:
        self.bootstrap(force_reindex=True)

    def _load_documents(self) -> Iterable[Document]:
        mode = self.config.knowledge.mode
        if mode == "prod":
            service_json = self.config.knowledge.google_service_account_json
            folder_id = self.config.knowledge.google_drive_folder_id
            if not service_json:
                raise RuntimeError("GoogleサービスアカウントのJSONパスが設定されていません。")
            if not folder_id:
                raise RuntimeError("Google DriveフォルダIDが設定されていません。")
            return GoogleDriveKnowledgeSource(
                service_account_json=service_json,
                folder_id=folder_id,
                faq_master_doc_id=self.config.knowledge.faq_master_doc_id,
            ).load_documents()
        # default: test/local mode
        if self.config.knowledge.knowledge_text_path:
            path = self.config.knowledge.knowledge_text_path
            return LocalMarkdownKnowledgeSource(path=path).load_documents()
        if self.config.knowledge.knowledge_text_inline:
            return LocalMarkdownKnowledgeSource(
                inline_text=self.config.knowledge.knowledge_text_inline
            ).load_documents()
        raise RuntimeError("テストモードでは KNOWLEDGE_TEXT_PATH もしくは KNOWLEDGE_TEXT を設定してください。")

    # ------------------------------------------------------------------
    def answer_question(
        self,
        *,
        question: str,
        user_id: str,
        channel_id: str,
    ) -> Answer:
        normalized = normalize_text(question)
        answer = self.memo_client.answer(normalized)
        now = datetime.now(timezone.utc)
        log_entry = QuestionLogEntry(
            timestamp=now,
            user_id=user_id,
            channel_id=channel_id,
            question=normalized,
            answer=answer.text,
            citations=answer.citations,
            score=answer.score,
            mode=self.config.knowledge.mode,
            escalated=answer.needs_escalation,
        )
        self.question_logger.append(log_entry)
        question_id = self._question_id(now, user_id, normalized)
        x, y = self._pseudo_embedding(normalized)
        topic = self._infer_topic(answer)
        status = "escalated" if answer.needs_escalation else "auto"
        self.embedding_store.add(
            EmbeddingRecord(
                question_id=question_id,
                timestamp=now,
                user_id=user_id,
                channel_id=channel_id,
                x=x,
                y=y,
                topic=topic,
                status=status,
            )
        )
        if answer.needs_escalation:
            payload = EscalationPayload(
                question=normalized,
                user_id=user_id,
                channel_id=channel_id,
                score=answer.score,
                citations=answer.citations,
                snippets=answer.snippets,
                reason="信頼度がしきい値を下回りました。",
            )
            self._record_escalation(now, payload)
            self._notify_escalation(payload)
        return answer

    # ------------------------------------------------------------------
    def _record_escalation(self, timestamp: datetime, payload: EscalationPayload) -> None:
        entry = EscalationEntry(
            timestamp=timestamp,
            user_id=payload.user_id,
            channel_id=payload.channel_id,
            question=payload.question,
            score=payload.score,
            top_snippets=payload.snippets,
            reason=payload.reason,
        )
        self.escalation_logger.append(entry)

    def _notify_escalation(self, payload: EscalationPayload) -> None:
        if self._escalation_callback:
            self._escalation_callback(payload)
        elif self.config.knowledge.mode == "test":
            print(
                "[ESCALATION]",
                payload.question,
                payload.user_id,
                payload.channel_id,
            )

    # ------------------------------------------------------------------
    def _question_id(self, timestamp: datetime, user_id: str, question: str) -> str:
        seed = f"{timestamp.isoformat()}|{user_id}|{question}".encode("utf-8")
        return hashlib.sha1(seed).hexdigest()

    def _pseudo_embedding(self, text: str) -> tuple[float, float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        x_int = int.from_bytes(digest[:8], "big")
        y_int = int.from_bytes(digest[8:16], "big")
        x = (x_int / 2**64) * 2 - 1
        y = (y_int / 2**64) * 2 - 1
        return float(x), float(y)

    def _infer_topic(self, answer: Answer) -> str:
        if answer.citations:
            return answer.citations[0].split("（", 1)[0]
        if answer.snippets:
            snippet = answer.snippets[0]
            return snippet[:20] + "..." if len(snippet) > 20 else snippet
        return "未分類"

    # ------------------------------------------------------------------
    def recent_questions(self, days: int = 7) -> List[dict]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        rows: List[dict] = []
        with self.config.dashboard.logs_file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    timestamp = datetime.fromisoformat(row["timestamp"])
                except (KeyError, ValueError):
                    continue
                if timestamp >= cutoff:
                    rows.append(
                        {
                            "timestamp": timestamp,
                            "user_id": row.get("user_id", ""),
                            "channel_id": row.get("channel_id", ""),
                            "question": row.get("question", ""),
                            "answer": row.get("answer", ""),
                            "citations": row.get("citations", "[]"),
                            "score": float(row.get("score", "0") or 0),
                            "mode": row.get("mode", ""),
                            "escalated": row.get("escalated", "0") == "1",
                        }
                    )
        return rows

    # ------------------------------------------------------------------
    def export_logs(self, *, user: str | None = None, start: datetime | None = None, end: datetime | None = None) -> List[QuestionLogEntry]:
        entries: List[QuestionLogEntry] = []
        with self.config.dashboard.logs_file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    timestamp = datetime.fromisoformat(row["timestamp"])
                except (KeyError, ValueError):
                    continue
                if start and timestamp < start:
                    continue
                if end and timestamp > end:
                    continue
                if user and row.get("user_id") != user:
                    continue
                entries.append(
                    QuestionLogEntry(
                        timestamp=timestamp,
                        user_id=row.get("user_id", ""),
                        channel_id=row.get("channel_id", ""),
                        question=row.get("question", ""),
                        answer=row.get("answer", ""),
                        citations=[],
                        score=float(row.get("score", "0") or 0),
                        mode=row.get("mode", ""),
                        escalated=row.get("escalated", "0") == "1",
                    )
                )
        return entries
