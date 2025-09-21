"""FastAPI application serving the dashboard and setup wizard."""
from __future__ import annotations

import csv
import io
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from qa_bot.config import AppConfig
from qa_bot.service import QABotService
from qa_bot.setup.manager import SetupManager
from qa_bot.utils.env import EnvironmentInfo, probe_env, select_pipeline

try:  # optional dependency for XLSX export
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional
    pd = None


TEMPLATES_DIR = Path(__file__).parent / "templates"


class DashboardData:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def load_logs(self) -> List[dict]:
        rows: List[dict] = []
        if not self.config.dashboard.logs_file.exists():
            return rows
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
                row["engine"] = row.get("engine", "rag")
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
        topic_counts: Dict[str, int] = {}
        for item in logs:
            topic = item.get("topic", "未分類")
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
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
        if not self.config.dashboard.embeddings_file.exists():
            return records
        store = self._load_embeddings()
        for item in store:
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
                [
                    "timestamp",
                    "user_id",
                    "channel_id",
                    "question",
                    "answer",
                    "score",
                    "mode",
                    "escalated",
                    "engine",
                ]
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
                        row.get("engine", "rag"),
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
                        "engine": row.get("engine", "rag"),
                    }
                    for row in filtered
                ]
            )
            buffer = io.BytesIO()
            df.to_excel(buffer, index=False)
            return buffer.getvalue()
        raise ValueError("未対応のフォーマットです")

    def _load_embeddings(self) -> List[dict]:
        if not self.config.dashboard.embeddings_file.exists():
            return []
        try:
            return pickle.loads(self.config.dashboard.embeddings_file.read_bytes())
        except Exception:
            return []


def create_app(
    config: AppConfig,
    service: Optional[QABotService] = None,
    env_info: Optional[EnvironmentInfo] = None,
    setup_manager: Optional[SetupManager] = None,
) -> FastAPI:
    env_info = env_info or probe_env()
    manager = setup_manager or SetupManager(config)
    data = DashboardData(config)
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app = FastAPI(title="QA Bot Dashboard", default_response_class=JSONResponse)

    def form_defaults() -> Dict[str, str]:
        env_values = manager.current_env()
        yaml_values = manager.current_yaml()
        rag_section = yaml_values.get("rag", {}) if isinstance(yaml_values.get("rag"), dict) else {}
        dash_section = (
            yaml_values.get("dashboard", {}) if isinstance(yaml_values.get("dashboard"), dict) else {}
        )
        setup_section = yaml_values.get("setup", {}) if isinstance(yaml_values.get("setup"), dict) else {}
        defaults = {
            "mode": env_values.get("MODE", setup_section.get("mode", config.knowledge.mode)),
            "rag_engine": rag_section.get("engine", config.rag.engine),
            "discord_token": env_values.get("DISCORD_BOT_TOKEN", ""),
            "discord_app_id": env_values.get("DISCORD_APP_ID", ""),
            "teacher_user_id": env_values.get("TEACHER_USER_ID", ""),
            "escalation_channel_id": env_values.get("ESCALATION_CHANNEL_ID", ""),
            "alerts_channel_id": env_values.get("ALERTS_CHANNEL_ID", ""),
            "gen_provider": env_values.get("GEN_PROVIDER", config.rag.gen_provider or ""),
            "gen_model": env_values.get("GEN_MODEL", config.rag.gen_model or ""),
            "openai_api_key": env_values.get("OPENAI_API_KEY", config.rag.openai_api_key or ""),
            "mem_model": env_values.get("MEM_MODEL", config.rag.mem_model),
            "ret_model": env_values.get("RET_MODEL", config.rag.ret_model),
            "knowledge_text_path": env_values.get(
                "KNOWLEDGE_TEXT_PATH",
                str(config.knowledge.knowledge_text_path or ""),
            ),
            "knowledge_text": env_values.get("KNOWLEDGE_TEXT", config.knowledge.knowledge_text_inline or ""),
            "google_service_account_json": env_values.get(
                "GOOGLE_SERVICE_ACCOUNT_JSON",
                str(config.knowledge.google_service_account_json or ""),
            ),
            "google_drive_folder_id": env_values.get("GOOGLE_DRIVE_FOLDER_ID", config.knowledge.google_drive_folder_id or ""),
            "faq_master_doc_id": env_values.get("FAQ_MASTER_DOC_ID", config.knowledge.faq_master_doc_id or ""),
            "retrieval_score_min": str(rag_section.get("retrieval_score_min", config.rag.retrieval_score_min)),
            "max_tokens": str(rag_section.get("max_tokens", config.rag.max_tokens)),
            "temperature": str(rag_section.get("temperature", config.rag.temperature)),
            "beacon_ratio": str(rag_section.get("beacon_ratio", config.rag.beacon_ratio)),
            "chunk_size": str(rag_section.get("chunk_size", config.rag.chunk_size)),
            "chunk_overlap": str(rag_section.get("chunk_overlap", config.rag.chunk_overlap)),
            "default_days": str(dash_section.get("default_days", config.dashboard.default_days)),
            "cluster_method": str(dash_section.get("cluster_method", config.dashboard.cluster_method)),
        }
        return defaults

    def setup_context(request: Request, message: str | None = None, error: str | None = None) -> dict:
        defaults = form_defaults()
        auto_engine = select_pipeline(env_info, None)
        current_engine = service.pipeline_name if service else auto_engine
        return {
            "request": request,
            "config": config,
            "env_info": env_info,
            "defaults": defaults,
            "message": message,
            "error": error,
            "auto_engine": auto_engine,
            "current_engine": current_engine,
            "setup_completed": config.setup.completed,
            "token_required": manager.requires_token(),
        }

    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request):
        if not config.setup.completed:
            return RedirectResponse(url="/setup")
        context = {
            "request": request,
            "config": config,
            "current_engine": service.pipeline_name if service else select_pipeline(env_info, config.rag.engine),
        }
        return templates.TemplateResponse("dashboard_home.html", context)

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "setup_completed": config.setup.completed,
            "selected_engine": service.pipeline_name if service else select_pipeline(env_info, config.rag.engine),
        }

    @app.get("/setup", response_class=HTMLResponse)
    async def get_setup(request: Request) -> HTMLResponse:
        return templates.TemplateResponse("setup.html", setup_context(request))

    @app.post("/setup/save", response_class=HTMLResponse)
    async def post_setup(request: Request) -> HTMLResponse:
        form = await request.form()
        token = form.get("access_token")
        if manager.requires_token() and not manager.validate_token(token):
            return templates.TemplateResponse(
                "setup.html",
                setup_context(request, error="アクセストークンが正しくありません。"),
                status_code=403,
            )
        env_updates = {
            "MODE": str(form.get("mode") or config.knowledge.mode),
            "LANG": config.lang,
            "DISCORD_BOT_TOKEN": form.get("discord_token", ""),
            "DISCORD_APP_ID": form.get("discord_app_id", ""),
            "TEACHER_USER_ID": form.get("teacher_user_id", ""),
            "ESCALATION_CHANNEL_ID": form.get("escalation_channel_id", ""),
            "ALERTS_CHANNEL_ID": form.get("alerts_channel_id", ""),
            "GEN_PROVIDER": form.get("gen_provider", ""),
            "GEN_MODEL": form.get("gen_model", ""),
            "OPENAI_API_KEY": form.get("openai_api_key", ""),
            "MEM_MODEL": form.get("mem_model", config.rag.mem_model),
            "RET_MODEL": form.get("ret_model", config.rag.ret_model),
            "KNOWLEDGE_TEXT_PATH": form.get("knowledge_text_path", ""),
            "KNOWLEDGE_TEXT": form.get("knowledge_text", ""),
            "GOOGLE_SERVICE_ACCOUNT_JSON": form.get("google_service_account_json", ""),
            "GOOGLE_DRIVE_FOLDER_ID": form.get("google_drive_folder_id", ""),
            "FAQ_MASTER_DOC_ID": form.get("faq_master_doc_id", ""),
        }

        def to_float(name: str, default: float) -> float:
            value = form.get(name)
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def to_int(name: str, default: int) -> int:
            value = form.get(name)
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        rag_settings = {
            "engine": str(form.get("rag_engine") or config.rag.engine),
            "retrieval_score_min": to_float("retrieval_score_min", config.rag.retrieval_score_min),
            "max_tokens": to_int("max_tokens", config.rag.max_tokens),
            "temperature": to_float("temperature", config.rag.temperature),
            "beacon_ratio": to_int("beacon_ratio", config.rag.beacon_ratio),
            "chunk_size": to_int("chunk_size", config.rag.chunk_size),
            "chunk_overlap": to_int("chunk_overlap", config.rag.chunk_overlap),
        }
        dashboard_settings = {
            "default_days": to_int("default_days", config.dashboard.default_days),
            "cluster_method": str(form.get("cluster_method") or config.dashboard.cluster_method),
        }
        setup_updates = {"mode": env_updates["MODE"]}
        result = manager.save(
            env_updates,
            rag_settings,
            dashboard_settings,
            setup_updates=setup_updates,
        )
        return templates.TemplateResponse(
            "setup.html",
            setup_context(request, message=result.message),
        )

    def _auth_check(access_token: Optional[str]) -> None:
        if manager.requires_token() and not manager.validate_token(access_token):
            raise HTTPException(status_code=403, detail="アクセストークンが無効です。")

    @app.post("/setup/test/discord")
    async def test_discord(payload: Dict[str, str] = Body(...)) -> dict:
        _auth_check(payload.get("access_token"))
        if not payload.get("token"):
            return {"ok": False, "message": "ボットトークンを入力してください。"}
        return {"ok": True, "message": "トークン形式の確認に成功しました。"}

    @app.post("/setup/test/google")
    async def test_google(payload: Dict[str, str] = Body(...)) -> dict:
        _auth_check(payload.get("access_token"))
        if not payload.get("service_json"):
            return {"ok": False, "message": "サービスアカウントJSONのパスを入力してください。"}
        if not Path(payload["service_json"]).expanduser().exists():
            return {"ok": False, "message": "指定されたファイルが見つかりません。"}
        return {"ok": True, "message": "ファイルパスの確認に成功しました。"}

    @app.post("/setup/test/llm")
    async def test_llm(payload: Dict[str, str] = Body(...)) -> dict:
        _auth_check(payload.get("access_token"))
        provider = payload.get("provider")
        model = payload.get("model")
        if not provider or not model:
            return {"ok": False, "message": "プロバイダとモデルを入力してください。"}
        if provider.lower() == "openai" and not payload.get("api_key"):
            return {"ok": False, "message": "OpenAIのAPIキーを入力してください。"}
        return {"ok": True, "message": "設定値の形式が正しいことを確認しました。"}

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
        media_type = (
            "text/csv"
            if fmt == "csv"
            else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        return StreamingResponse(
            io.BytesIO(payload),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    return app

