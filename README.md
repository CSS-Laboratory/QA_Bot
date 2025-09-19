# 日本語対応・講義用 QA ボット

このリポジトリは Discord 上で動作する日本語対応の講義用 Q\&A ボットのリファレンス実装です。MemoRAG を利用した検索・生成型回答、未知質問のエスカレーション、ログ収集と可視化ダッシュボードを備えています。

## 主なコンポーネント

- `qa_bot/config.py` — `.env` から設定を読み込み、必要なディレクトリを初期化します。
- `qa_bot/service.py` — 知識ソースの読み込み、メモリ構築、質問処理、ログ保存、エスカレーション通知を統合します。
- `qa_bot/rag/memorag_client.py` — MemoRAG へのラッパー。ライブラリが利用できない場合は簡易的な n-gram ベースのフォールバック検索を提供します。
- `qa_bot/knowledge/` — テストモード用のローカル Markdown 読み込み (`local.py`) と、本番用 Google Drive 連携 (`google_docs.py`) を実装します。
- `qa_bot/discord/bot.py` — Discord ボット本体。メンションや DM からの質問応答、各種スラッシュコマンド、エスカレーション通知を担当します。
- `qa_bot/dashboard/app.py` — FastAPI ベースのダッシュボード API。統計サマリー、質問マップ、CSV/XLSX エクスポートを提供します。
- `data/knowledge_test.md` — `MODE=test` で利用するサンプルの知識ベースです。

## セットアップ

1. Python 3.11 環境を用意し、必要に応じて仮想環境を作成します。
2. `pip install -r requirements.txt` もしくは `pyproject.toml` に従い依存関係を導入します。MemoRAG や Google API を利用する場合は GPU や追加ライブラリが必要です。
3. `.env` を作成し、以下の最低限の値を設定します。

   ```env
   MODE=test
   LANG=ja
   CACHE_DIR=./data/cache
   KNOWLEDGE_TEXT_PATH=./data/knowledge_test.md
   DISCORD_BOT_TOKEN=...  # bot を起動する場合
   DISCORD_APP_ID=...
   TEACHER_USER_ID=1234567890
   ESCALATION_CHANNEL_ID=1234567890
   ALERTS_CHANNEL_ID=1234567890
   GEN_PROVIDER=openai
   GEN_MODEL=gpt-5-nano
   OPENAI_API_KEY=sk-...
   MEM_MODEL=TommyChien/memorag-qwen2-7b-inst
   RET_MODEL=BAAI/bge-m3
   RETRIEVAL_SCORE_MIN=0.35
   ```

4. 本番運用 (`MODE=prod`) では Google Drive/Docs 用のサービスアカウント JSON、ドキュメント ID を追加します。

## 実行方法

### Discord ボットのみ

```bash
python main.py bot
```

### ダッシュボードのみ

```bash
python main.py dashboard --host 0.0.0.0 --port 8000
```

### ボットとダッシュボードを同時起動

```bash
python main.py both --host 0.0.0.0 --port 8000
```

## テストモードについて

`MODE=test` では `data/knowledge_test.md` から知識を読み込み、FAISS や外部 API を利用せずにフォールバック実装で応答を生成します。未知質問は `data/escalations.csv` に記録され、コンソールに通知されます。

## ログとダッシュボード

- 質問ログ: `data/logs/questions.csv`
- エスカレーションログ: `data/escalations.csv`
- 埋め込みキャッシュ: `data/pickle/embeddings.pkl`

FastAPI の `/metrics/summary` で統計サマリー、`/metrics/map` で 2D マップ、`/export` で CSV/XLSX を取得できます。

## 注意事項

- Google API や Discord API を利用する場合は、各サービスのレート制限と利用規約を遵守してください。
- MemoRAG を利用する際は GPU ドライバや `faiss` のセットアップが必要です。フォールバックモードは軽量ですが簡易的なスコアリングのみ行います。
- エラーハンドリングや再試行は `QABotService` と Discord ボット内で実装されていますが、運用環境では追加の監視を推奨します。
