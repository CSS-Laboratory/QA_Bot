"""Utilities for the web setup wizard."""
from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Dict, Optional

from qa_bot.config import AppConfig

try:  # pragma: no cover - optional dependency
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


@dataclass(slots=True)
class SetupResult:
    success: bool
    message: str


class SetupManager:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.env_path = config.env_path
        self.config_path = config.config_path
        self._token: Optional[str] = None
        if not config.setup.completed and not config.admin_password:
            self._token = secrets.token_urlsafe(12)
            print("[Setup] 初期設定トークン:", self._token)

    # ------------------------------------------------------------------
    @property
    def token(self) -> Optional[str]:
        return self._token

    def requires_token(self) -> bool:
        return bool(self.config.admin_password or self._token)

    def validate_token(self, supplied: Optional[str]) -> bool:
        if self.config.admin_password:
            return supplied == self.config.admin_password
        if self._token:
            return supplied == self._token
        return True

    # ------------------------------------------------------------------
    def current_env(self) -> Dict[str, str]:
        return self.config.load_env_values()

    def current_yaml(self) -> Dict[str, Dict[str, object]]:
        if not self.config_path.exists() or yaml is None:
            return {}
        try:
            data = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        return data

    # ------------------------------------------------------------------
    def save(
        self,
        env_updates: Dict[str, str],
        rag_settings: Dict[str, object],
        dashboard_settings: Dict[str, object],
        *,
        setup_updates: Optional[Dict[str, object]] = None,
        mark_completed: bool = True,
    ) -> SetupResult:
        self._write_env(env_updates)
        self._write_yaml(
            rag_settings,
            dashboard_settings,
            setup_updates=setup_updates,
            mark_completed=mark_completed,
        )
        if mark_completed:
            self.config.setup.completed = True
            self._token = None
        return SetupResult(success=True, message="設定を保存しました。再起動してください。")

    # ------------------------------------------------------------------
    def _write_env(self, updates: Dict[str, str]) -> None:
        existing = self.current_env()
        merged = {**existing, **{k: v for k, v in updates.items() if v is not None}}
        lines = [f"{key}={value}" for key, value in sorted(merged.items())]
        self.env_path.parent.mkdir(parents=True, exist_ok=True)
        self.env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_yaml(
        self,
        rag_settings: Dict[str, object],
        dashboard_settings: Dict[str, object],
        *,
        setup_updates: Optional[Dict[str, object]],
        mark_completed: bool,
    ) -> None:
        if yaml is None:
            return
        data = self.current_yaml()
        rag = data.get("rag", {}) if isinstance(data.get("rag"), dict) else {}
        dash = data.get("dashboard", {}) if isinstance(data.get("dashboard"), dict) else {}
        setup = data.get("setup", {}) if isinstance(data.get("setup"), dict) else {}
        rag.update(rag_settings)
        dash.update(dashboard_settings)
        if setup_updates:
            setup.update(setup_updates)
        if mark_completed:
            setup["completed"] = True
        data.update({"rag": rag, "dashboard": dash, "setup": setup})
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=True)

