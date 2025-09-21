from __future__ import annotations

from pathlib import Path

import pytest

from qa_bot.config import AppConfig
from qa_bot.service import QABotService


@pytest.fixture()
def temp_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    knowledge = tmp_path / "knowledge.md"
    knowledge.write_text(
        """# FAQ\n\n## Q. 宿題の提出方法は？\nLMSに提出してください。""",
        encoding="utf-8",
    )
    cache_dir = tmp_path / "cache"
    data_dir = tmp_path / "data"
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "MODE=test",
                "LANG=ja",
                f"CACHE_DIR={cache_dir}",
                f"KNOWLEDGE_TEXT_PATH={knowledge}",
                "DISCORD_BOT_TOKEN=",
                "DISCORD_APP_ID=0",
            ]
        ),
        encoding="utf-8",
    )
    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text(
        """
setup:
  completed: true
rag:
  engine: rag
  retrieval_score_min: 0.05
  chunk_size: 900
  chunk_overlap: 120
dashboard:
  default_days: 7
  cluster_method: hdbscan
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("QA_BOT_ENV_PATH", str(env_file))
    monkeypatch.setenv("QA_BOT_CONFIG_PATH", str(config_yaml))
    monkeypatch.setenv("MODE", "test")
    monkeypatch.setenv("KNOWLEDGE_TEXT_PATH", str(knowledge))
    monkeypatch.setenv("CACHE_DIR", str(cache_dir))
    config = AppConfig.from_env()
    config.dashboard.logs_file = data_dir / "logs" / "questions.csv"
    config.dashboard.escalations_file = data_dir / "escalations.csv"
    config.dashboard.embeddings_file = data_dir / "embeddings.pkl"
    config.ensure_directories()
    return config


def test_service_answers_question(temp_config: AppConfig) -> None:
    service = QABotService(temp_config)
    service.bootstrap(force_reindex=True)
    answer = service.answer_question(
        question="宿題の提出方法を教えてください。",
        user_id="student1",
        channel_id="channel1",
    )
    assert answer.engine == "rag"
    assert "提出" in answer.text
    assert answer.citations
    assert not answer.needs_escalation
    assert temp_config.dashboard.logs_file.exists()
