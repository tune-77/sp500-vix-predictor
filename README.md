# S&P 500 価格予測（LSTM）

- **データ**: yfinance で ^GSPC（S&P 500）と ^VIX（VIX）の過去10年
- **特徴量**: 終値 + VIX、MinMaxScaler(0–1)
- **モデル**: TensorFlow/Keras LSTM（60日 → 翌日終値）、Dropout、EarlyStopping
- **出力**: RMSE、実測 vs 予測のグラフ（Matplotlib / Plotly）、`forecast_results.csv` に追記

## 初回（学習済みモデルを作る）

```bash
cd sp500_forecast
pip install -r requirements.txt
python predict.py --train
```

これで `model.keras` と `scaler_params.npz` が作成されます。  
**これらをリポジトリにコミットしておくと、GitHub Actions では「予測だけ」が走り、無料枠を節約できます。**

## 日次予測のみ（モデル読み込み）

```bash
python predict.py --predict-only
# または CI 環境では CI=true で自動的に予測のみ
```

## 通知（任意）

- **LINE**: リポジトリの Secrets に `LINE_NOTIFY_TOKEN` を設定
- **Discord**: `DISCORD_WEBHOOK_URL` を設定

GitHub Actions の job で同じ名前の Secrets を渡すと、予測完了時に通知されます。

## スケジュール

- `.github/workflows/daily_predict.yml`: 毎日 **日本時間 9:00**（UTC 0:00）に実行
- 実行後に `forecast_results.csv` を自動 commit & push
