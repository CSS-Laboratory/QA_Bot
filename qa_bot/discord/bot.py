"""Discord bot implementation for the QA assistant."""
from __future__ import annotations

import asyncio
import io
from datetime import datetime
from typing import List, Optional

import discord
from discord import app_commands
from discord.ext import commands

from qa_bot.config import AppConfig
from qa_bot.rag.pipelines import Answer
from qa_bot.service import EscalationPayload, QABotService


class QADiscordBot(commands.Bot):
    def __init__(self, config: AppConfig, service: QABotService) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents)
        self.config = config
        self.service = service
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._pending_escalations: List[EscalationPayload] = []

    async def setup_hook(self) -> None:
        self._loop = asyncio.get_running_loop()
        self.service.set_escalation_callback(self._queue_escalation)
        self.tree.add_command(self.help_command_impl)
        self.tree.add_command(self.status_command)
        self.tree.add_command(self.reindex_command)
        self.tree.add_command(self.export_command)
        self.tree.add_command(self.set_faq_doc)
        self.tree.add_command(self.set_drive_folder)
        await self.tree.sync()

    async def on_ready(self) -> None:
        if self._pending_escalations:
            for payload in list(self._pending_escalations):
                await self._send_escalation(payload)
            self._pending_escalations.clear()
        await super().on_ready()

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return
        if isinstance(message.channel, discord.DMChannel) or self.user in message.mentions:
            await self._handle_question(message)
        await super().on_message(message)

    async def _handle_question(self, message: discord.Message) -> None:
        await message.trigger_typing()
        answer: Answer = await asyncio.to_thread(
            self.service.answer_question,
            question=message.content,
            user_id=str(message.author.id),
            channel_id=str(message.channel.id),
        )
        await message.reply(answer.text)
        if answer.needs_escalation and self.config.discord.teacher_user_id:
            await message.reply(
                f"<@{self.config.discord.teacher_user_id}> への確認が必要な質問として記録しました。"
            )

    # ----------------------------- slash commands -----------------------------
    @app_commands.command(name="help", description="利用方法を表示します")
    async def help_command_impl(self, interaction: discord.Interaction) -> None:
        text = (
            "日本語で質問を送信すると自動で知識ベースから回答します。\n"
            "/reindex でインデックス再構築、/export でログ出力が可能です。"
        )
        await interaction.response.send_message(text, ephemeral=True)

    @app_commands.command(name="status", description="現在の状態を表示します")
    async def status_command(self, interaction: discord.Interaction) -> None:
        doc_count = len(self.service.documents)
        recent = self.service.recent_questions(days=7)
        escalated = sum(1 for r in recent if r["escalated"])
        text = (
            f"モード: {self.config.knowledge.mode}\n"
            f"RAGエンジン: {self.service.pipeline_name}\n"
            f"ドキュメント数: {doc_count}\n"
            f"直近7日の質問数: {len(recent)} (エスカレーション {escalated})"
        )
        await interaction.response.send_message(text, ephemeral=True)

    @app_commands.command(name="reindex", description="知識ベースを再構築します")
    async def reindex_command(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        await asyncio.to_thread(self.service.rebuild_memory)
        await interaction.followup.send("インデックスを再構築しました。", ephemeral=True)

    @app_commands.command(name="export", description="質問ログをエクスポートします")
    @app_commands.describe(
        target="user または all",
        user_id="target=user の場合に指定する Discord ユーザーID",
        date_range="YYYY-MM-DD:YYYY-MM-DD",
    )
    async def export_command(
        self,
        interaction: discord.Interaction,
        target: str,
        user_id: Optional[str] = None,
        date_range: Optional[str] = None,
    ) -> None:
        if target not in {"user", "all"}:
            await interaction.response.send_message("target には user もしくは all を指定してください。", ephemeral=True)
            return
        if target == "user" and not user_id:
            await interaction.response.send_message("user を指定する場合は user_id を入力してください。", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        user_filter = user_id if target == "user" else None
        start = end = None
        if date_range:
            try:
                start_str, end_str = date_range.split(":", 1)
                start = datetime.fromisoformat(start_str)
                end = datetime.fromisoformat(end_str)
            except ValueError:
                await interaction.followup.send("日付範囲の形式が正しくありません。", ephemeral=True)
                return
        entries = await asyncio.to_thread(
            self.service.export_logs, user=user_filter, start=start, end=end
        )
        lines = [
            "timestamp,user_id,channel_id,question,answer,score,mode,escalated",
        ]
        for entry in entries:
            lines.append(
                ",".join(
                    [
                        entry.timestamp.isoformat(),
                        entry.user_id,
                        entry.channel_id,
                        entry.question.replace(",", " "),
                        entry.answer.replace(",", " "),
                        f"{entry.score:.4f}",
                        entry.mode,
                        "1" if entry.escalated else "0",
                    ]
                )
            )
        content = "\n".join(lines)
        buffer = io.BytesIO(content.encode("utf-8"))
        buffer.seek(0)
        file = discord.File(buffer, filename="export.csv")
        await interaction.followup.send(
            content="CSVファイルを添付しました。",
            ephemeral=True,
            file=file,
        )

    @app_commands.command(name="set_faq_doc", description="FAQドキュメントIDを更新します")
    async def set_faq_doc(self, interaction: discord.Interaction, doc_id: str) -> None:
        self.config.knowledge.faq_master_doc_id = doc_id
        self.service.state.set("faq_master_doc_id", doc_id)
        await interaction.response.send_message("FAQドキュメントIDを更新しました。", ephemeral=True)

    @app_commands.command(name="set_drive_folder", description="DriveフォルダIDを更新します")
    async def set_drive_folder(self, interaction: discord.Interaction, folder_id: str) -> None:
        self.config.knowledge.google_drive_folder_id = folder_id
        self.service.state.set("google_drive_folder_id", folder_id)
        await interaction.response.send_message("DriveフォルダIDを更新しました。", ephemeral=True)

    # ----------------------------- escalation handling -----------------------------
    def _queue_escalation(self, payload: EscalationPayload) -> None:
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._send_escalation(payload), self._loop)
        else:
            self._pending_escalations.append(payload)

    async def _send_escalation(self, payload: EscalationPayload) -> None:
        channel_id = self.config.discord.escalation_channel_id
        teacher_id = self.config.discord.teacher_user_id
        if not channel_id or not teacher_id:
            return
        channel = self.get_channel(channel_id)
        if not channel:
            return
        citation_text = "\n".join(payload.citations) or "該当する引用がありません"
        snippet_text = "\n".join(payload.snippets[:3]) or "スニペットなし"
        message = (
            f"<@{teacher_id}> 教員確認が必要な質問です。\n"
            f"質問: {payload.question}\n"
            f"スコア: {payload.score:.3f}\n"
            f"引用: {citation_text}\n"
            f"スニペット:\n{snippet_text}"
        )
        await channel.send(message)
