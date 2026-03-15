# CLAUDE.md — S&P 500 VIX Predictor

## プロジェクト概要

S&P 500（^GSPC）の翌日終値をLSTMモデルで予測するシステム。
VIX指数を追加特徴量として活用し、Claude Agent SDKによる自然言語インターフェースも提供。

## ファイル構成

| ファイル | 役割 |
|---|---|
| `predict.py` | LSTMモデルの学習・予測・CSV出力 |
| `agent.py` | Claude Agent SDKによる対話型S&P500分析エージェント |
| `requirements.txt` | 依存パッケージ一覧 |
| `model.keras` | 学習済みLSTMモデル（初回学習後に生成） |
| `scaler_params.npz` | MinMaxScalerのパラメータ（初回学習後に生成） |
| `forecast_results.csv` | 予測結果の蓄積CSV |

## セットアップ

```bash
pip install -r requirements.txt
```

## よく使うコマンド

```bash
# LSTMモデルの学習（初回 or 再学習）
python predict.py --train

# 予測のみ実行（学習済みモデルが必要）
python predict.py --predict-only

# S&P500分析エージェント（デフォルト：今後1週間の予測）
python agent.py

# エージェントに具体的な質問
python agent.py --prompt "現在のS&P500のテクニカル分析をしてください"

# 対話モード
python agent.py --interactive
```

## アーキテクチャ

### predict.py（LSTMモデル）
- **データ**: yfinance で過去10年分の ^GSPC・^VIX を取得
- **特徴量**: 終値 + VIX（2次元）、MinMaxScaler(0–1)正規化
- **モデル**: LSTM(64) → Dropout(0.3) → LSTM(32) → Dropout(0.3) → Dense(16) → Dense(1)
- **学習**: 60日シーケンス → 翌日終値予測、EarlyStopping(patience=15)
- **出力**: `forecast_results.csv` に予測日・予測終値・実行日時を追記

### agent.py（Claude Agent SDK）
- **モデル**: `claude-opus-4-6`（adaptive thinking有効）
- **ツール**: WebSearch・WebFetch・Bash・Read
- **役割**: テクニカル/ファンダメンタルズ/VIX分析の専門エージェント
- **機能**: 最新市場データの収集 → LSTM予測結果との統合 → 総合分析レポート生成

## 環境変数

| 変数名 | 用途 | 必須 |
|---|---|---|
| `ANTHROPIC_API_KEY` | Claude API認証（agent.py用） | agent.py使用時 |
| `LINE_NOTIFY_TOKEN` | LINE通知（任意） | 任意 |
| `DISCORD_WEBHOOK_URL` | Discord通知（任意） | 任意 |

## 開発ガイドライン

- `predict.py` の変更時は `--train` で再学習して動作確認すること
- `agent.py` の変更時は `python agent.py --verbose` でログを確認
- スケーラーとモデルのパスは `BASE_DIR` 相対で管理（`MODEL_PATH`, `SCALER_PATH`）
- CIでは `CI=true` 環境変数で自動的に `--predict-only` モードで動作する

## CI/CD

GitHub Actions（`.github/workflows/daily_predict.yml`）にて毎日 JST 9:00 に実行：
1. `python predict.py --predict-only` で予測
2. `forecast_results.csv` を自動コミット＆プッシュ
3. 通知Webhookが設定されていれば通知を送信
