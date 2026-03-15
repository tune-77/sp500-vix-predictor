#!/usr/bin/env python3
"""
S&P 500 分析エージェント（Gemini API版）
Google Gemini API を使用した S&P 500 予測・分析の専門エージェント

使い方:
    export GEMINI_API_KEY="your-api-key"
    python agent.py
    python agent.py --prompt "明日のS&P500を予測してください"
    python agent.py --interactive
"""

import argparse
import os
import sys
import json
from pathlib import Path

import yfinance as yf
import pandas as pd
import numpy as np
from google import genai
from google.genai import types

BASE_DIR = Path(__file__).resolve().parent

SYSTEM_PROMPT = """あなたはS&P 500（SPX）分析の世界クラスのエキスパートです。
以下の専門知識を持っています：

【専門分野】
- テクニカル分析: 移動平均線(SMA/EMA)、RSI、MACD、ボリンジャーバンド、フィボナッチ水準、出来高分析
- ファンダメンタルズ分析: 企業業績、PER/PBR、マクロ経済指標（GDP、インフレ、雇用）
- VIX（恐怖指数）と市場センチメントの解読
- FRB金融政策（利上げ/利下げサイクル）の相場への影響
- セクターローテーション分析（エネルギー・テック・金融など）
- 季節性パターン（1月効果、セル・イン・メイなど）
- LSTMを含む機械学習モデルによる価格予測

【行動指針】
1. 利用可能なツールで最新データを取得してから分析を行う
2. 複数の視点（テクニカル・ファンダメンタルズ・センチメント）を統合して総合判断する
3. 予測には必ず根拠と信頼度（低/中/高）を明示する
4. リスク要因（下落シナリオ）も必ず提示する
5. 日本語で回答する（専門用語は英語のまま可）
6. 数値・パーセント・日付は具体的に示す

【免責事項】
分析結果は情報提供目的であり、投資助言ではありません。投資判断はご自身の責任で行ってください。"""

# ──────────────────────────────────────────────
# ツール定義（Gemini Function Calling）
# ──────────────────────────────────────────────

def get_sp500_data(days: int = 30) -> str:
    """S&P500とVIXの最新価格データを取得する"""
    try:
        end = pd.Timestamp.now()
        start = end - pd.Timedelta(days=days)
        sp = yf.download("^GSPC", start=start, end=end, progress=False, auto_adjust=True)
        vix = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)
        if sp.empty:
            return json.dumps({"error": "S&P500データの取得に失敗しました"})

        latest_sp = float(sp["Close"].iloc[-1])
        prev_sp = float(sp["Close"].iloc[-2]) if len(sp) > 1 else latest_sp
        change = latest_sp - prev_sp
        change_pct = (change / prev_sp) * 100

        result = {
            "sp500": {
                "latest_close": round(latest_sp, 2),
                "prev_close": round(prev_sp, 2),
                "change": round(change, 2),
                "change_pct": round(change_pct, 2),
                "high_30d": round(float(sp["Close"].max()), 2),
                "low_30d": round(float(sp["Close"].min()), 2),
                "sma_20": round(float(sp["Close"].tail(20).mean()), 2),
                "date": str(sp.index[-1].date()),
            }
        }
        if not vix.empty:
            result["vix"] = {
                "latest": round(float(vix["Close"].iloc[-1]), 2),
                "avg_30d": round(float(vix["Close"].mean()), 2),
            }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_technical_indicators(period: str = "3mo") -> str:
    """テクニカル指標（RSI・MACD・ボリンジャーバンド）を計算して返す"""
    try:
        sp = yf.download("^GSPC", period=period, progress=False, auto_adjust=True)
        if sp.empty:
            return json.dumps({"error": "データ取得失敗"})

        close = sp["Close"].squeeze()

        # RSI (14)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        rsi = float((100 - 100 / (1 + rs)).iloc[-1])

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd_line = float((ema12 - ema26).iloc[-1])
        signal_line = float((ema12 - ema26).ewm(span=9).mean().iloc[-1])

        # ボリンジャーバンド (20)
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper_band = float((sma20 + 2 * std20).iloc[-1])
        lower_band = float((sma20 - 2 * std20).iloc[-1])
        current_price = float(close.iloc[-1])

        # 200日移動平均
        sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

        return json.dumps({
            "rsi_14": round(rsi, 1),
            "rsi_signal": "買われ過ぎ" if rsi > 70 else "売られ過ぎ" if rsi < 30 else "中立",
            "macd": round(macd_line, 2),
            "macd_signal": round(signal_line, 2),
            "macd_histogram": round(macd_line - signal_line, 2),
            "bollinger_upper": round(upper_band, 2),
            "bollinger_lower": round(lower_band, 2),
            "bollinger_position_pct": round(
                (current_price - lower_band) / (upper_band - lower_band) * 100, 1
            ),
            "sma_200": round(sma200, 2) if sma200 else None,
            "above_sma200": current_price > sma200 if sma200 else None,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_lstm_forecast() -> str:
    """保存済みLSTMモデルで翌日のS&P500終値を予測する"""
    model_path = BASE_DIR / "model.keras"
    scaler_path = BASE_DIR / "scaler_params.npz"
    if not model_path.exists() or not scaler_path.exists():
        return json.dumps({
            "status": "model_not_found",
            "message": "LSTMモデル未学習。`python predict.py --train` で学習してください。"
        })
    try:
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras.models import load_model

        df = yf.download(["^GSPC", "^VIX"], period="1y", progress=False, auto_adjust=True)
        close = df["Close"]["^GSPC"].rename("Close")
        vix = df["Close"]["^VIX"].rename("VIX")
        combined = pd.concat([close, vix], axis=1).dropna()

        d = np.load(scaler_path, allow_pickle=True)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.min_ = d["min_"]
        scaler.scale_ = d["scale_"]

        scaled = scaler.transform(combined[["Close", "VIX"]])
        LOOKBACK = 60
        if len(scaled) < LOOKBACK:
            return json.dumps({"error": "データ不足"})
        seq = scaled[-LOOKBACK:].reshape(1, LOOKBACK, 2)

        model = load_model(model_path)
        pred_scaled = float(model.predict(seq, verbose=0).flatten()[0])
        pred_price = float(
            scaler.inverse_transform([[pred_scaled, 0.0]])[0][0]
        )
        current = float(combined["Close"].iloc[-1])
        return json.dumps({
            "status": "success",
            "current_price": round(current, 2),
            "predicted_next_day": round(pred_price, 2),
            "predicted_change": round(pred_price - current, 2),
            "predicted_change_pct": round((pred_price - current) / current * 100, 2),
            "as_of": str(combined.index[-1].date()),
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ──────────────────────────────────────────────
# Gemini ツールスキーマ
# ──────────────────────────────────────────────

TOOLS = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="get_sp500_data",
            description="S&P500とVIXの最新価格・直近N日の高値・安値・20日移動平均を取得する",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "days": types.Schema(
                        type=types.Type.INTEGER,
                        description="取得する日数（デフォルト30日）",
                    )
                },
            ),
        ),
        types.FunctionDeclaration(
            name="get_technical_indicators",
            description="テクニカル指標（RSI14・MACD・ボリンジャーバンド・SMA200）を計算して返す",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "period": types.Schema(
                        type=types.Type.STRING,
                        description="取得期間 例: '1mo' '3mo' '6mo' '1y'",
                    )
                },
            ),
        ),
        types.FunctionDeclaration(
            name="get_lstm_forecast",
            description="保存済みLSTMモデルで翌日のS&P500終値を予測する",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
            ),
        ),
    ])
]

TOOL_MAP = {
    "get_sp500_data": get_sp500_data,
    "get_technical_indicators": get_technical_indicators,
    "get_lstm_forecast": get_lstm_forecast,
}

# ──────────────────────────────────────────────
# エージェントループ
# ──────────────────────────────────────────────

def run_agent(prompt: str, verbose: bool = False) -> str:
    """Gemini APIでS&P500分析エージェントを実行（ツールループ付き）"""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("エラー: GEMINI_API_KEY が設定されていません。", file=sys.stderr)
        print("  export GEMINI_API_KEY='your-api-key'", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=TOOLS,
        temperature=0.7,
    )

    contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]

    # ツール呼び出しループ
    for turn in range(10):
        if verbose:
            print(f"[ターン {turn + 1}] Gemini APIに送信中...", flush=True)

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
            config=config,
        )

        candidate = response.candidates[0]
        contents.append(types.Content(role="model", parts=candidate.content.parts))

        # ツール呼び出しがあれば実行
        tool_calls = [p for p in candidate.content.parts if p.function_call]
        if not tool_calls:
            break  # テキスト応答のみ → 完了

        tool_results = []
        for part in tool_calls:
            fc = part.function_call
            fn = TOOL_MAP.get(fc.name)
            if fn is None:
                result_str = json.dumps({"error": f"未知のツール: {fc.name}"})
            else:
                args = dict(fc.args) if fc.args else {}
                if verbose:
                    print(f"  [ツール呼び出し] {fc.name}({args})", flush=True)
                result_str = fn(**args)
                if verbose:
                    print(f"  [ツール結果] {result_str[:200]}...", flush=True)

            tool_results.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        response={"result": result_str},
                    )
                )
            )

        contents.append(types.Content(role="user", parts=tool_results))

    # 最終テキスト抽出
    final_text = ""
    for part in candidate.content.parts:
        if hasattr(part, "text") and part.text:
            final_text += part.text
    return final_text


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def run_interactive(verbose: bool = False):
    print("=" * 60)
    print("  S&P 500 分析エージェント（Gemini API版）")
    print("  終了するには 'exit' または 'quit' と入力")
    print("=" * 60)
    while True:
        print()
        try:
            user_input = input("質問 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "終了"):
            print("終了します。")
            break
        print("\n分析中...\n")
        result = run_agent(user_input, verbose=verbose)
        print("\n" + "=" * 60)
        print(result)
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="S&P 500 分析エージェント（Gemini API版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  export GEMINI_API_KEY="your-key"
  python agent.py
  python agent.py --prompt "今週のS&P500の主要サポートとレジスタンスは？"
  python agent.py --interactive
        """,
    )
    parser.add_argument("--prompt", "-p", type=str, help="質問・指示")
    parser.add_argument("--interactive", "-i", action="store_true", help="対話モード")
    parser.add_argument("--verbose", "-v", action="store_true", help="詳細ログ")
    args = parser.parse_args()

    if args.interactive:
        run_interactive(verbose=args.verbose)
        return

    prompt = args.prompt or (
        "現在のS&P 500の状況を分析し、今後1週間の値動きを予測してください。"
        "最新の市場データ、VIX、主要テクニカル指標、LSTMモデル予測を考慮した上で、"
        "強気・弱気それぞれのシナリオと重要な価格水準を教えてください。"
    )

    print(f"プロンプト: {prompt}\n分析中...\n")
    result = run_agent(prompt, verbose=args.verbose)
    print("\n" + "=" * 60)
    print("【分析結果】")
    print("=" * 60)
    print(result)
    print("=" * 60)


if __name__ == "__main__":
    main()
