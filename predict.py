#!/usr/bin/env python3
"""
S&P 500 (^GSPC) 価格予測モデル
- 特徴量: 終値 + VIX指数、MinMaxScaler、LSTM(60日→翌日)
- 学習済みモデルがあれば読み込んで予測のみ実行（CI用）
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# TensorFlow is optional for predict-only when using saved model + precomputed scaler
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# --- 設定 ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.keras"
SCALER_PATH = BASE_DIR / "scaler_params.npz"
FORECAST_CSV = BASE_DIR / "forecast_results.csv"
LOOKBACK = 60
TICKER = "^GSPC"
VIX_TICKER = "^VIX"
YEARS = 10


def fetch_data(years: int = YEARS) -> pd.DataFrame:
    """S&P 500 と VIX の過去データを取得"""
    end = pd.Timestamp.now().normalize()
    start = end - pd.DateOffset(years=years)
    sp = yf.download(TICKER, start=start, end=end, progress=False, auto_adjust=True)
    vix = yf.download(VIX_TICKER, start=start, end=end, progress=False, auto_adjust=True)
    if sp.empty or vix.empty:
        raise RuntimeError("Failed to fetch data from yfinance")
    sp = sp["Close"].rename("Close")
    vix = vix["Close"].rename("VIX")
    df = pd.concat([sp, vix], axis=1).dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def build_sequences(df: pd.DataFrame, scaler: MinMaxScaler, lookback: int):
    """終値+VIXをスケーリングし、lookback日分のシーケンスと翌日終値ラベルを作成"""
    scaled = scaler.transform(df[["Close", "VIX"]])
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback : i])
        y.append(scaled[i, 0])  # 翌日終値（スケール済み）
    return np.array(X), np.array(y)


def build_model(shape: tuple) -> keras.Model:
    """LSTM + Dropout モデル"""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=shape),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_and_save(df: pd.DataFrame) -> tuple:
    """学習してモデルとスケーラーを保存。RMSEと履歴を返す"""
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required for training")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df[["Close", "VIX"]])
    X, y = build_sequences(df, scaler, LOOKBACK)
    # 時系列のため最後からテスト用に分割（例: 最後20%）
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    np.savez(
        SCALER_PATH,
        min_=scaler.min_,
        scale_=scaler.scale_,
        feature_range_min=0.0,
        feature_range_max=1.0,
    )
    model = build_model((X.shape[1], X.shape[2]))
    early = EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early],
        verbose=0,
    )
    model.save(MODEL_PATH)
    pred = model.predict(X_test, verbose=0).flatten()
    # 逆変換は終値列のみ（スケーラーは [Close, VIX] の2列）
    pred_inv = scaler.inverse_transform(np.column_stack([pred, np.zeros_like(pred)]))[:, 0]
    y_test_inv = scaler.inverse_transform(np.column_stack([y_test, np.zeros_like(y_test)]))[:, 0]
    rmse = np.sqrt(mean_squared_error(y_test_inv, pred_inv))
    split = int(len(X) * 0.8)
    test_dates = df.index[split + LOOKBACK : split + LOOKBACK + len(y_test_inv)]
    return rmse, history, df, scaler, X_test, pred_inv, y_test_inv, test_dates


def load_scaler() -> MinMaxScaler:
    """保存したスケーラーを復元"""
    d = np.load(SCALER_PATH, allow_pickle=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.min_ = d["min_"]
    scaler.scale_ = d["scale_"]
    return scaler


def predict_only() -> pd.DataFrame:
    """保存済みモデルで最新データから翌日終値を予測し、CSVに追記"""
    df = fetch_data(years=YEARS)
    scaler = load_scaler()
    X, _ = build_sequences(df, scaler, LOOKBACK)
    if len(X) == 0:
        raise RuntimeError("Not enough data for prediction (need at least LOOKBACK+1 rows)")
    model = load_model(MODEL_PATH)
    last_seq = X[-1 :]
    pred_scaled = model.predict(last_seq, verbose=0).flatten()[0]
    pred_inv = scaler.inverse_transform(
        np.column_stack([np.array([pred_scaled]), np.array([0.0])])
    )[0, 0]
    as_of = pd.Timestamp.now().normalize()
    # 翌営業日は簡易的に +1 日（実際は祝日考慮すると複雑なのでここでは翌日）
    forecast_date = as_of + pd.Timedelta(days=1)
    row = {
        "forecast_date": forecast_date.strftime("%Y-%m-%d"),
        "predicted_close": round(pred_inv, 2),
        "as_of_date": as_of.strftime("%Y-%m-%d"),
        "created_at": pd.Timestamp.now().isoformat(),
    }
    result_df = pd.DataFrame([row])
    if FORECAST_CSV.exists():
        existing = pd.read_csv(FORECAST_CSV)
        result_df = pd.concat([existing, result_df], ignore_index=True)
    result_df.to_csv(FORECAST_CSV, index=False)
    return result_df


def run_full_pipeline(do_plot: bool = True, do_plotly: bool = True) -> float:
    """訓練から評価・プロットまで一括実行（モデルがない場合）"""
    df = fetch_data(years=YEARS)
    rmse, history, df, scaler, X_test, pred_inv, y_test_inv, test_dates = train_and_save(df)
    print(f"Test RMSE: {rmse:.2f}")
    if do_plot:
        plt.figure(figsize=(12, 5))
        plt.plot(test_dates, y_test_inv, label="Actual", alpha=0.8)
        plt.plot(test_dates, pred_inv, label="Predicted", alpha=0.8)
        plt.legend()
        plt.title("S&P 500: Actual vs Predicted (Test Set)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(BASE_DIR / "comparison.png", dpi=150)
        plt.close()
        print(f"Saved {BASE_DIR / 'comparison.png'}")
    if do_plotly and PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_dates, y=y_test_inv, name="Actual", mode="lines"))
        fig.add_trace(go.Scatter(x=test_dates, y=pred_inv, name="Predicted", mode="lines"))
        fig.update_layout(
            title="S&P 500: Actual vs Predicted (Interactive)",
            xaxis_title="Date",
            yaxis_title="Close",
            template="plotly_white",
        )
        fig.write_html(BASE_DIR / "comparison_plotly.html")
        print(f"Saved {BASE_DIR / 'comparison_plotly.html'}")
    return rmse


def main():
    parser = argparse.ArgumentParser(description="S&P 500 LSTM prediction")
    parser.add_argument("--train", action="store_true", help="Force retrain and save model")
    parser.add_argument("--no-plot", action="store_true", help="Skip saving comparison plots")
    parser.add_argument("--no-plotly", action="store_true", help="Skip Plotly HTML")
    parser.add_argument("--predict-only", action="store_true", help="Load model, predict, append CSV only")
    args = parser.parse_args()
    ci = os.environ.get("CI", "").lower() in ("1", "true", "yes")

    # CI または --predict-only: モデルがあれば予測のみ
    if ci or args.predict_only:
        if not MODEL_PATH.exists() or not SCALER_PATH.exists():
            print("No saved model/scaler found. Run without --predict-only (and not in CI) to train first.", file=sys.stderr)
            sys.exit(1)
        result = predict_only()
        print("Prediction appended to forecast_results.csv:")
        print(result.tail(1).to_string(index=False))
        # 通知: 環境変数で Webhook を設定すると送信可能
        notify_if_configured(result.tail(1))
        return

    # モデルがなければ学習してから予測；--train なら常に再学習
    if args.train or not MODEL_PATH.exists() or not SCALER_PATH.exists():
        if not TF_AVAILABLE:
            print("TensorFlow is required for training.", file=sys.stderr)
            sys.exit(1)
        run_full_pipeline(do_plot=not args.no_plot, do_plotly=not args.no_plotly)
    else:
        # モデルあり: 予測のみ実行してCSV追記
        result = predict_only()
        print("Prediction appended to forecast_results.csv:")
        print(result.tail(1).to_string(index=False))
        notify_if_configured(result.tail(1))

    # 追記後、今回の予測を表示
    if FORECAST_CSV.exists():
        latest = pd.read_csv(FORECAST_CSV).tail(5)
        print("\nLatest forecast_results.csv (last 5 rows):")
        print(latest.to_string(index=False))


def notify_if_configured(latest_row: pd.DataFrame):
    """LINE / Discord Webhook が設定されていれば通知（任意）"""
    line_token = os.environ.get("LINE_NOTIFY_TOKEN")
    discord_webhook = os.environ.get("DISCORD_WEBHOOK_URL")
    if not latest_row.empty:
        msg = (
            f"S&P 500 予測完了\n"
            f"予測日: {latest_row['forecast_date'].iloc[0]}\n"
            f"予測終値: {latest_row['predicted_close'].iloc[0]}"
        )
    else:
        return
    if line_token:
        try:
            import urllib.request
            import urllib.parse
            req = urllib.request.Request(
                "https://notify-api.line.me/api/notify",
                data=urllib.parse.urlencode({"message": msg}).encode(),
                headers={"Authorization": f"Bearer {line_token}"},
                method="POST",
            )
            urllib.request.urlopen(req)
        except Exception as e:
            print(f"LINE notify failed: {e}", file=sys.stderr)
    if discord_webhook:
        try:
            import urllib.request
            import json
            body = json.dumps({"content": msg}).encode()
            req = urllib.request.Request(
                discord_webhook,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req)
        except Exception as e:
            print(f"Discord notify failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
