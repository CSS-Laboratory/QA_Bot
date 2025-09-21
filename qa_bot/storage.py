"""Persistent storage utilities for logs and cached state."""
from __future__ import annotations

import csv
import json
import pickle
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List


@dataclass(slots=True)
class QuestionLogEntry:
    timestamp: datetime
    user_id: str
    channel_id: str
    question: str
    answer: str
    citations: List[str]
    score: float
    mode: str
    escalated: bool
    engine: str

    def to_csv_row(self) -> List[str]:
        return [
            self.timestamp.isoformat(),
            self.user_id,
            self.channel_id,
            self.question,
            self.answer,
            json.dumps(self.citations, ensure_ascii=False),
            f"{self.score:.4f}",
            self.mode,
            "1" if self.escalated else "0",
            self.engine,
        ]


class QuestionLogger:
    headers = [
        "timestamp",
        "user_id",
        "channel_id",
        "question",
        "answer",
        "citations",
        "score",
        "mode",
        "escalated",
        "engine",
    ]

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            with self.file_path.open("w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(self.headers)

    def append(self, entry: QuestionLogEntry) -> None:
        with self.file_path.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(entry.to_csv_row())


@dataclass(slots=True)
class EscalationEntry:
    timestamp: datetime
    user_id: str
    channel_id: str
    question: str
    score: float
    top_snippets: List[str]
    reason: str

    def to_csv_row(self) -> List[str]:
        return [
            self.timestamp.isoformat(),
            self.user_id,
            self.channel_id,
            self.question,
            f"{self.score:.4f}",
            json.dumps(self.top_snippets, ensure_ascii=False),
            self.reason,
        ]


class EscalationLogger:
    headers = [
        "timestamp",
        "user_id",
        "channel_id",
        "question",
        "score",
        "top_snippets",
        "reason",
    ]

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            with self.file_path.open("w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(self.headers)

    def append(self, entry: EscalationEntry) -> None:
        with self.file_path.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(entry.to_csv_row())


class PickleState:
    """Simple key-value store backed by pickle."""

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> dict:
        if not self.file_path.exists():
            return {}
        try:
            return pickle.loads(self.file_path.read_bytes())
        except Exception:
            return {}

    def save(self) -> None:
        self.file_path.write_bytes(pickle.dumps(self._data))

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def set(self, key: str, value) -> None:
        self._data[key] = value
        self.save()


@dataclass(slots=True)
class EmbeddingRecord:
    question_id: str
    timestamp: datetime
    user_id: str
    channel_id: str
    x: float
    y: float
    topic: str
    status: str


class EmbeddingStore:
    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._records = self._load()

    def _load(self) -> List[dict]:
        if not self.file_path.exists():
            return []
        try:
            return pickle.loads(self.file_path.read_bytes())
        except Exception:
            return []

    def add(self, record: EmbeddingRecord) -> None:
        self._records.append(asdict(record))
        self._save()

    def _save(self) -> None:
        self.file_path.write_bytes(pickle.dumps(self._records))

    def all(self) -> List[dict]:
        return list(self._records)
