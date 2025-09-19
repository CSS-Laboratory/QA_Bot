"""FastAPI dashboard exposing analytics for the QA bot."""
from __future__ import annotations

import csv
import io
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse

from qa_bot.config import AppConfig
from qa_bot.storage import EmbeddingStore

try:  # optional dependency for XLSX export
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional
    pd = None


class DashboardData:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.embedding_store = EmbeddingStore(config.dashboard.embeddings_file)

    def load_logs(self) -> List[dict]:
        rows: List[dict] = []
        with self.config.dashboard.logs_file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    timestamp = datetime.fromisoformat(row["timestamp"])
                except Exception:
                    continue
                row["timestamp"] = timestamp
                row["escalated"] = row.get("escalated", "0") == "1"
                row["score"] = float(row.get("score", "0") or 0)
                rows.append(row)
        return rows

    def summary(self) -> dict:
        logs = self.load_logs()
        now = datetime.now(timezone.utc)
        last_week_start = now - timedelta(days=7)
        prev_week_start = now - timedelta(days=14)
        last_week = [row for row in logs if row["timestamp"] >= last_week_start]
        prev_week = [row for row in logs if prev_week_start <= row["timestamp"] < last_week_start]
        last_count = len(last_week)
        prev_count = len(prev_week) or 1
        delta = last_count - len(prev_week)
        auto_count = sum(1 for row in last_week if not row["escalated"])
        escalated_count = last_count - auto_count
        topic_counts = Counter(rec.get("topic", "未分類") for rec in self.embedding_store.all())
        top_topics = topic_counts.most_common(5)
        return {
            "recent_count": last_count,
            "previous_count": len(prev_week),
            "difference": delta,
            "auto_rate": auto_count / last_count if last_count else 0,
            "escalation_rate": escalated_count / last_count if last_count else 0,
            "top_topics": top_topics,
        }

    def map_data(
        self,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        topic: Optional[str] = None,
        user: Optional[str] = None,
        channel: Optional[str] = None,
    ) -> List[dict]:
        records = []
        for item in self.embedding_store.all():
            timestamp = item.get("timestamp")
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except ValueError:
                    continue
            if start and timestamp < start:
                continue
            if end and timestamp > end:
                continue
            if topic and item.get("topic") != topic:
                continue
            if user and item.get("user_id") != user:
                continue
            if channel and item.get("channel_id") != channel:
                continue
            records.append(
                {
                    "question_id": item.get("question_id"),
                    "timestamp": timestamp.isoformat(),
                    "user_id": item.get("user_id"),
                    "channel_id": item.get("channel_id"),
                    "x": item.get("x"),
                    "y": item.get("y"),
                    "topic": item.get("topic"),
                    "status": item.get("status"),
                }
            )
        return records

    def export(
        self,
        *,
        user: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        fmt: str = "csv",
    ) -> bytes:
        logs = self.load_logs()
        filtered = []
        for row in logs:
            if user and row.get("user_id") != user:
                continue
            if start and row["timestamp"] < start:
                continue
            if end and row["timestamp"] > end:
                continue
            filtered.append(row)
        if fmt == "csv":
            buffer = io.StringIO()
            writer = csv.writer(buffer)
            writer.writerow(
                ["timestamp", "user_id", "channel_id", "question", "answer", "score", "mode", "escalated"]
            )
            for row in filtered:
                writer.writerow(
                    [
                        row["timestamp"].isoformat(),
                        row.get("user_id", ""),
                        row.get("channel_id", ""),
                        row.get("question", ""),
                        row.get("answer", ""),
                        row.get("score", 0),
                        row.get("mode", ""),
                        "1" if row.get("escalated") else "0",
                    ]
                )
            return buffer.getvalue().encode("utf-8")
        if fmt == "xlsx":
            if pd is None:
                raise RuntimeError("pandasがインストールされていません")
            df = pd.DataFrame(
                [
                    {
                        "timestamp": row["timestamp"],
                        "user_id": row.get("user_id", ""),
                        "channel_id": row.get("channel_id", ""),
                        "question": row.get("question", ""),
                        "answer": row.get("answer", ""),
                        "score": row.get("score", 0),
                        "mode": row.get("mode", ""),
                        "escalated": row.get("escalated", False),
                    }
                    for row in filtered
                ]
            )
            buffer = io.BytesIO()
            df.to_excel(buffer, index=False)
            return buffer.getvalue()
        raise ValueError("未対応のフォーマットです")


def create_app(config: AppConfig) -> FastAPI:
    data = DashboardData(config)
    app = FastAPI(title="QA Bot Dashboard", default_response_class=JSONResponse)

    @app.get("/metrics/summary")
    def get_summary() -> dict:
        return data.summary()

    @app.get("/metrics/map")
    def get_map(
        start: Optional[datetime] = Query(None),
        end: Optional[datetime] = Query(None),
        topic: Optional[str] = Query(None),
        user: Optional[str] = Query(None),
        channel: Optional[str] = Query(None),
    ) -> List[dict]:
        return data.map_data(start=start, end=end, topic=topic, user=user, channel=channel)

    @app.get("/export")
    def export(
        fmt: str = Query("csv", pattern="^(csv|xlsx)$"),
        user: Optional[str] = Query(None),
        start: Optional[datetime] = Query(None),
        end: Optional[datetime] = Query(None),
    ):
        try:
            payload = data.export(user=user, start=start, end=end, fmt=fmt)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        filename = f"export.{fmt}"
        media_type = "text/csv" if fmt == "csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        return StreamingResponse(io.BytesIO(payload), media_type=media_type, headers={"Content-Disposition": f"attachment; filename={filename}"})

    return app
