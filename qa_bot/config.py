"""Configuration handling for the QA bot."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv(*args, **kwargs):
        return False


load_dotenv()


@dataclass(slots=True)
class DiscordConfig:
    token: str
    app_id: Optional[int]
    teacher_user_id: Optional[int]
    escalation_channel_id: Optional[int]
    alerts_channel_id: Optional[int]


@dataclass(slots=True)
class KnowledgeConfig:
    mode: str
    knowledge_text_path: Optional[Path]
    knowledge_text_inline: Optional[str]
    google_drive_folder_id: Optional[str]
    faq_master_doc_id: Optional[str]
    google_service_account_json: Optional[Path]
    cache_dir: Path


@dataclass(slots=True)
class RagConfig:
    mem_model: str
    ret_model: str
    gen_provider: Optional[str]
    gen_model: Optional[str]
    openai_api_key: Optional[str]
    beacon_ratio: int
    retrieval_score_min: float
    max_tokens: int
    temperature: float


@dataclass(slots=True)
class DashboardConfig:
    data_root: Path
    logs_file: Path
    escalations_file: Path
    embeddings_file: Path


@dataclass(slots=True)
class AppConfig:
    discord: DiscordConfig
    knowledge: KnowledgeConfig
    rag: RagConfig
    dashboard: DashboardConfig
    lang: str = "ja"

    @staticmethod
    def from_env() -> "AppConfig":
        mode = os.getenv("MODE", "test").lower()
        cache_dir = Path(os.getenv("CACHE_DIR", "./data/cache"))
        data_root = Path("./data")
        logs_dir = data_root / "logs"
        cache_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        (data_root / "pickle").mkdir(parents=True, exist_ok=True)

        return AppConfig(
            discord=DiscordConfig(
                token=os.getenv("DISCORD_BOT_TOKEN", ""),
                app_id=int(os.getenv("DISCORD_APP_ID")) if os.getenv("DISCORD_APP_ID") else None,
                teacher_user_id=int(os.getenv("TEACHER_USER_ID")) if os.getenv("TEACHER_USER_ID") else None,
                escalation_channel_id=int(os.getenv("ESCALATION_CHANNEL_ID")) if os.getenv("ESCALATION_CHANNEL_ID") else None,
                alerts_channel_id=int(os.getenv("ALERTS_CHANNEL_ID")) if os.getenv("ALERTS_CHANNEL_ID") else None,
            ),
            knowledge=KnowledgeConfig(
                mode=mode,
                knowledge_text_path=Path(os.getenv("KNOWLEDGE_TEXT_PATH", "")).resolve() if os.getenv("KNOWLEDGE_TEXT_PATH") else None,
                knowledge_text_inline=os.getenv("KNOWLEDGE_TEXT"),
                google_drive_folder_id=os.getenv("GOOGLE_DRIVE_FOLDER_ID"),
                faq_master_doc_id=os.getenv("FAQ_MASTER_DOC_ID"),
                google_service_account_json=Path(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")).resolve() if os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON") else None,
                cache_dir=cache_dir,
            ),
            rag=RagConfig(
                mem_model=os.getenv("MEM_MODEL", "TommyChien/memorag-qwen2-7b-inst"),
                ret_model=os.getenv("RET_MODEL", "BAAI/bge-m3"),
                gen_provider=os.getenv("GEN_PROVIDER"),
                gen_model=os.getenv("GEN_MODEL"),
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                beacon_ratio=int(os.getenv("BEACON_RATIO", "4")),
                retrieval_score_min=float(os.getenv("RETRIEVAL_SCORE_MIN", "0.35")),
                max_tokens=int(os.getenv("MAX_TOKENS", "256")),
                temperature=float(os.getenv("TEMPERATURE", "0.2")),
            ),
            dashboard=DashboardConfig(
                data_root=data_root,
                logs_file=logs_dir / "questions.csv",
                escalations_file=data_root / "escalations.csv",
                embeddings_file=data_root / "pickle" / "embeddings.pkl",
            ),
        )

    def ensure_directories(self) -> None:
        self.knowledge.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dashboard.logs_file.parent.mkdir(parents=True, exist_ok=True)
        self.dashboard.escalations_file.parent.mkdir(parents=True, exist_ok=True)
        self.dashboard.embeddings_file.parent.mkdir(parents=True, exist_ok=True)
