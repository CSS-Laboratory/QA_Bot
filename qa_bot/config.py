"""Configuration handling for the QA bot with .env and YAML support."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import os

try:  # pragma: no cover - optional dependency at runtime
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv, dotenv_values
except Exception:  # pragma: no cover
    def load_dotenv(*args, **kwargs):
        return False

    def dotenv_values(*args, **kwargs):  # type: ignore
        return {}


CONFIG_FILE_ENV = "QA_BOT_CONFIG_PATH"
ENV_FILE_ENV = "QA_BOT_ENV_PATH"


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
    engine: str
    chunk_size: int
    chunk_overlap: int


@dataclass(slots=True)
class DashboardConfig:
    data_root: Path
    logs_file: Path
    escalations_file: Path
    embeddings_file: Path
    default_days: int
    cluster_method: str


@dataclass(slots=True)
class SetupConfig:
    completed: bool


@dataclass(slots=True)
class AppConfig:
    discord: DiscordConfig
    knowledge: KnowledgeConfig
    rag: RagConfig
    dashboard: DashboardConfig
    setup: SetupConfig
    lang: str
    config_path: Path
    env_path: Path
    admin_password: Optional[str]

    @staticmethod
    def from_env() -> "AppConfig":
        env_path = Path(os.getenv(ENV_FILE_ENV, ".env")).resolve()
        config_path = Path(os.getenv(CONFIG_FILE_ENV, "config.yaml")).resolve()

        if env_path.exists():
            load_dotenv(env_path)
        else:  # pragma: no cover - fallback when .env missing
            load_dotenv()

        yaml_data: Dict[str, Any] = {}
        if config_path.exists() and yaml is not None:
            try:
                yaml_data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            except Exception:  # pragma: no cover - ignore invalid YAML
                yaml_data = {}

        rag_section = yaml_data.get("rag", {}) if isinstance(yaml_data, dict) else {}
        dashboard_section = yaml_data.get("dashboard", {}) if isinstance(yaml_data, dict) else {}
        setup_section = yaml_data.get("setup", {}) if isinstance(yaml_data, dict) else {}

        def _int(value: Any, default: int) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def _float(value: Any, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        mode = os.getenv("MODE", str(setup_section.get("mode", "test"))).lower()
        cache_dir = Path(os.getenv("CACHE_DIR", "./data/cache")).resolve()
        data_root = Path("./data").resolve()
        logs_dir = data_root / "logs"

        rag_engine = str(rag_section.get("engine", "auto")).lower()
        if rag_engine not in {"auto", "memorag", "rag"}:
            rag_engine = "auto"

        rag_chunk_size = _int(rag_section.get("chunk_size"), _int(os.getenv("CHUNK_SIZE"), 900))
        rag_chunk_overlap = _int(rag_section.get("chunk_overlap"), _int(os.getenv("CHUNK_OVERLAP"), 120))
        default_days = _int(dashboard_section.get("default_days"), 7)
        cluster_method = str(dashboard_section.get("cluster_method", "hdbscan"))

        retrieval_score_min = _float(
            rag_section.get("retrieval_score_min"),
            _float(os.getenv("RETRIEVAL_SCORE_MIN"), 0.35),
        )
        max_tokens = _int(rag_section.get("max_tokens"), _int(os.getenv("MAX_TOKENS"), 256))
        temperature = _float(rag_section.get("temperature"), _float(os.getenv("TEMPERATURE"), 0.2))
        beacon_ratio = _int(rag_section.get("beacon_ratio"), _int(os.getenv("BEACON_RATIO"), 4))

        knowledge_text_path = os.getenv("KNOWLEDGE_TEXT_PATH")
        google_sa_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

        config = AppConfig(
            discord=DiscordConfig(
                token=os.getenv("DISCORD_BOT_TOKEN", ""),
                app_id=int(os.getenv("DISCORD_APP_ID")) if os.getenv("DISCORD_APP_ID") else None,
                teacher_user_id=int(os.getenv("TEACHER_USER_ID")) if os.getenv("TEACHER_USER_ID") else None,
                escalation_channel_id=int(os.getenv("ESCALATION_CHANNEL_ID")) if os.getenv("ESCALATION_CHANNEL_ID") else None,
                alerts_channel_id=int(os.getenv("ALERTS_CHANNEL_ID")) if os.getenv("ALERTS_CHANNEL_ID") else None,
            ),
            knowledge=KnowledgeConfig(
                mode=mode,
                knowledge_text_path=Path(knowledge_text_path).resolve() if knowledge_text_path else None,
                knowledge_text_inline=os.getenv("KNOWLEDGE_TEXT"),
                google_drive_folder_id=os.getenv("GOOGLE_DRIVE_FOLDER_ID"),
                faq_master_doc_id=os.getenv("FAQ_MASTER_DOC_ID"),
                google_service_account_json=Path(google_sa_json).resolve() if google_sa_json else None,
                cache_dir=cache_dir,
            ),
            rag=RagConfig(
                mem_model=os.getenv("MEM_MODEL", "TommyChien/memorag-qwen2-7b-inst"),
                ret_model=os.getenv("RET_MODEL", "BAAI/bge-m3"),
                gen_provider=os.getenv("GEN_PROVIDER"),
                gen_model=os.getenv("GEN_MODEL"),
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                beacon_ratio=beacon_ratio,
                retrieval_score_min=retrieval_score_min,
                max_tokens=max_tokens,
                temperature=temperature,
                engine=rag_engine,
                chunk_size=rag_chunk_size,
                chunk_overlap=rag_chunk_overlap,
            ),
            dashboard=DashboardConfig(
                data_root=data_root,
                logs_file=logs_dir / "questions.csv",
                escalations_file=data_root / "escalations.csv",
                embeddings_file=data_root / "pickle" / "embeddings.pkl",
                default_days=default_days,
                cluster_method=cluster_method,
            ),
            setup=SetupConfig(completed=bool(setup_section.get("completed", False))),
            lang=os.getenv("LANG", "ja"),
            config_path=config_path,
            env_path=env_path,
            admin_password=os.getenv("ADMIN_PASSWORD"),
        )
        return config

    def ensure_directories(self) -> None:
        self.knowledge.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dashboard.logs_file.parent.mkdir(parents=True, exist_ok=True)
        self.dashboard.escalations_file.parent.mkdir(parents=True, exist_ok=True)
        self.dashboard.embeddings_file.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

    def load_env_values(self) -> Dict[str, str]:
        values = dotenv_values(self.env_path) if self.env_path.exists() else {}
        return {k: str(v) for k, v in values.items() if v is not None}

