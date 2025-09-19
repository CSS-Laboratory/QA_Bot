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
    logs_dir = tmp_path / "data" / "logs"
    monkeypatch.setenv("MODE", "test")
    monkeypatch.setenv("KNOWLEDGE_TEXT_PATH", str(knowledge))
    monkeypatch.setenv("CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "")
    monkeypatch.setenv("DISCORD_APP_ID", "0")
    monkeypatch.setenv("RETRIEVAL_SCORE_MIN", "0.2")
    config = AppConfig.from_env()
    config.dashboard.logs_file = logs_dir / "questions.csv"
    config.dashboard.escalations_file = tmp_path / "data" / "escalations.csv"
    config.dashboard.embeddings_file = tmp_path / "data" / "embeddings.pkl"
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
    assert "提出" in answer.text
    assert answer.citations
    assert not answer.needs_escalation
    assert temp_config.dashboard.logs_file.exists()
