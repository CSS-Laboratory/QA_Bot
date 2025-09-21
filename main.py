"""Entry point for running the QA bot and dashboard."""
from __future__ import annotations

import argparse
import asyncio
from typing import Literal, Optional

import uvicorn

from qa_bot.config import AppConfig
from qa_bot.dashboard.app import create_app
from qa_bot.discord.bot import QADiscordBot
from qa_bot.service import QABotService
from qa_bot.setup.manager import SetupManager
from qa_bot.utils.env import EnvironmentInfo, probe_env


async def run_discord_bot(config: AppConfig, service: QABotService) -> None:
    if not config.discord.token:
        raise RuntimeError("DISCORD_BOT_TOKEN が設定されていません。")
    bot = QADiscordBot(config, service)
    await bot.start(config.discord.token)


async def run_dashboard(
    config: AppConfig,
    host: str,
    port: int,
    *,
    service: Optional[QABotService],
    setup_manager: SetupManager,
    env_info: EnvironmentInfo,
) -> None:
    app = create_app(config, service=service, env_info=env_info, setup_manager=setup_manager)
    server = uvicorn.Server(
        uvicorn.Config(app, host=host, port=port, log_level="info")
    )
    await server.serve()


async def run_all(
    config: AppConfig,
    service: QABotService,
    host: str,
    port: int,
    *,
    setup_manager: SetupManager,
    env_info: EnvironmentInfo,
) -> None:
    app = create_app(config, service=service, env_info=env_info, setup_manager=setup_manager)
    server = uvicorn.Server(
        uvicorn.Config(app, host=host, port=port, loop="asyncio", lifespan="on")
    )
    bot = QADiscordBot(config, service)
    await asyncio.gather(bot.start(config.discord.token), server.serve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QA Bot runner")
    parser.add_argument(
        "target",
        choices=["bot", "dashboard", "both"],
        help="起動するコンポーネント",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig.from_env()
    env_info = probe_env()
    setup_manager = SetupManager(config)
    service: Optional[QABotService] = None
    if config.setup.completed:
        service = QABotService(config, env_info=env_info)
        service.bootstrap()
    else:
        try:
            service = QABotService(config, env_info=env_info)
            service.bootstrap()
        except Exception as exc:
            print("[Setup] インデックス初期化は保留:", exc)
    target: Literal["bot", "dashboard", "both"] = args.target
    if target in {"bot", "both"}:
        if not config.setup.completed:
            raise RuntimeError("設定ウィザードを完了するまでDiscordボットは起動できません。")
        if not config.discord.token:
            raise RuntimeError("DISCORD_BOT_TOKEN が設定されていません。")
        if not service:
            raise RuntimeError("サービスの初期化に失敗しました。")
    if target == "bot":
        asyncio.run(run_discord_bot(config, service))
    elif target == "dashboard":
        asyncio.run(
            run_dashboard(
                config,
                args.host,
                args.port,
                service=service,
                setup_manager=setup_manager,
                env_info=env_info,
            )
        )
    else:
        if not service:
            raise RuntimeError("サービスの初期化に失敗しました。")
        asyncio.run(
            run_all(
                config,
                service,
                args.host,
                args.port,
                setup_manager=setup_manager,
                env_info=env_info,
            )
        )


if __name__ == "__main__":
    main()
