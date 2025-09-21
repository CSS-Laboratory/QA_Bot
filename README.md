# 日本語対応・講義用 QA ボット

このリポジトリは Discord 上で動作する日本語対応の講義用 Q\&A ボットのリファレンス実装です。GPU と MemoRAG が利用できる環境では高速なメモリ検索パイプラインを使用し、条件を満たさない場合は自動的に標準RAG（FAISS + BGE-M3）にフォールバックします。初回起動時は Web ベースの設定ウィザードが立ち上がり、必要な API キーや Discord チャンネル ID を GUI で登録できます。

## 主なコンポーネント

- `qa_bot/config.py` — `.env` と `config.yaml` を読み込み、RAG エンジンの強制指定やチャンク設定を含むアプリ設定を構築します。
- `qa_bot/service.py` — 環境診断に基づき MemoRAG / 標準RAG を切り替え、知識ソースのロード、回答生成、ログ保存、エスカレーション通知を統合します。
- `qa_bot/rag/pipelines.py` — MemoRAG パイプラインと標準RAGパイプラインを実装し、引用付き日本語回答を返します。
- `qa_bot/knowledge/` — テストモード用のローカル Markdown 読み込み (`local.py`) と、本番用 Google Drive 連携 (`google_docs.py`) を実装します。
- `qa_bot/discord/bot.py` — Discord ボット本体。メンションや DM からの質問応答、各種スラッシュコマンド、エスカレーション通知を担当します。
- `qa_bot/dashboard/app.py` — FastAPI ベースのダッシュボード＆設定ウィザード。環境診断の表示、設定保存、統計サマリー、質問マップ、CSV/XLSX エクスポートを提供します。
- `qa_bot/setup/manager.py` — 設定ウィザードから `.env` と `config.yaml` を安全に更新します。
- `data/knowledge_test.md` — `MODE=test` で利用するサンプルの知識ベースです。

## セットアップ

1. Python 3.11 環境を用意し、必要に応じて仮想環境を作成します。
2. `pip install -r requirements.txt` で依存関係を導入します。GPU で MemoRAG を利用する場合は `torch` + `faiss-gpu` が必要です（条件を満たさない場合は自動で CPU RAG に切り替わります）。
3. 初回起動は設定ウィザードを使うため、下記コマンドでダッシュボードを起動します。

   ```bash
   python main.py dashboard --host 127.0.0.1 --port 8000
   ```

4. コンソールに表示される一時トークン（もしくは `ADMIN_PASSWORD`）を使って `http://127.0.0.1:8000/setup` にアクセスし、モード・RAGエンジン・Discord / LLM / Google 各種キーを入力して保存します。保存時に `.env` と `config.yaml` が生成され、以降はウィザードなしで起動できます。

5. 本番運用 (`MODE=prod`) ではサービスアカウント JSON、`GOOGLE_DRIVE_FOLDER_ID`、`FAQ_MASTER_DOC_ID` を設定してください。

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

## RAG エンジンの自動切替

起動時に `probe_env()` で以下を確認します。

1. `torch.cuda.is_available()` が真であること
2. `memorag` の import に成功すること
3. `faiss.get_num_gpus() >= 1` を満たすこと

条件を満たす場合は MemoRAG パイプラインが選択され、それ以外の場合は CPU ベースの標準RAGが利用されます。設定ウィザードから `AUTO / MemoRAG / RAG` を強制指定することも可能です。

## テストモードについて

`MODE=test` では `data/knowledge_test.md` またはウィザードで入力した Markdown テキストを知識源として利用します。未知質問は `data/escalations.csv` に記録され、Discord 設定が未入力の場合はコンソールへエスカレーション通知を出力します。

## ログとダッシュボード

- 質問ログ: `data/logs/questions.csv`
- エスカレーションログ: `data/escalations.csv`
- 埋め込みキャッシュ: `data/pickle/embeddings.pkl`

FastAPI の `/metrics/summary` で統計サマリー、`/metrics/map` で 2D マップ、`/export` で CSV/XLSX を取得できます。`/setup` から設定の再編集も可能です。

## 注意事項

- Google API や Discord API を利用する場合は、各サービスのレート制限と利用規約を遵守してください。
- MemoRAG を利用する際は GPU ドライバや `faiss-gpu` のセットアップが必要です。標準RAGは CPU 上で動作し、BGE-M3 埋め込みと FAISS/内積検索で回答を生成します。
- エラーハンドリングや再試行は `QABotService` と Discord ボット内で実装されていますが、運用環境では追加の監視を推奨します。
