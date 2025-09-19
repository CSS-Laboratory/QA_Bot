"""Entry point for running the QA bot and dashboard."""
from __future__ import annotations

import argparse
import asyncio
from typing import Literal

import uvicorn

from qa_bot.config import AppConfig
from qa_bot.dashboard.app import create_app
from qa_bot.discord.bot import QADiscordBot
from qa_bot.service import QABotService


async def run_discord_bot(config: AppConfig, service: QABotService) -> None:
    if not config.discord.token:
        raise RuntimeError("DISCORD_BOT_TOKEN が設定されていません。")
    bot = QADiscordBot(config, service)
    await bot.start(config.discord.token)


async def run_dashboard(config: AppConfig, host: str, port: int) -> None:
    app = create_app(config)
    server = uvicorn.Server(
        uvicorn.Config(app, host=host, port=port, log_level="info")
    )
    await server.serve()


async def run_all(config: AppConfig, service: QABotService, host: str, port: int) -> None:
    app = create_app(config)
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
    service = QABotService(config)
    service.bootstrap()
    target: Literal["bot", "dashboard", "both"] = args.target
    if target in {"bot", "both"} and not config.discord.token:
        raise RuntimeError("DISCORD_BOT_TOKEN が設定されていません。")
    if target == "bot":
        asyncio.run(run_discord_bot(config, service))
    elif target == "dashboard":
        asyncio.run(run_dashboard(config, args.host, args.port))
    else:
        asyncio.run(run_all(config, service, args.host, args.port))


if __name__ == "__main__":
    main()
