#!/usr/bin/env python3
"""
S&P 500 分析エージェント（Claude Agent SDK版）
Claude Agent SDK を使用した S&P 500 予測・分析の専門エージェント

使い方:
    export ANTHROPIC_API_KEY="your-api-key"
    python agent.py
    python agent.py --prompt "明日のS&P500を予測してください"
    python agent.py --interactive
"""

import argparse
import os
import sys
from pathlib import Path

import anyio
from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

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

【利用可能なツール】
- WebSearch: 最新の市場ニュース・経済指標を検索
- WebFetch: 特定URLのコンテンツを取得・分析
- Bash: Pythonスクリプトの実行（LSTM予測: `python predict.py --predict-only`）
- Read: ローカルファイルの読み込み（forecast_results.csv など）

【行動指針】
1. 利用可能なツールで最新データを取得してから分析を行う
2. LSTMモデルの予測結果は `python predict.py --predict-only` または `forecast_results.csv` から取得する
3. 複数の視点（テクニカル・ファンダメンタルズ・センチメント）を統合して総合判断する
4. 予測には必ず根拠と信頼度（低/中/高）を明示する
5. リスク要因（下落シナリオ）も必ず提示する
6. 日本語で回答する（専門用語は英語のまま可）
7. 数値・パーセント・日付は具体的に示す

【免責事項】
分析結果は情報提供目的であり、投資助言ではありません。投資判断はご自身の責任で行ってください。"""

DEFAULT_PROMPT = (
    "現在のS&P 500の状況を分析し、今後1週間の値動きを予測してください。"
    "最新の市場データ、VIX、主要テクニカル指標、LSTMモデル予測を考慮した上で、"
    "強気・弱気それぞれのシナリオと重要な価格水準を教えてください。"
)


async def run_agent(prompt: str, verbose: bool = False) -> str:
    """Claude Agent SDKでS&P500分析エージェントを実行"""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("エラー: ANTHROPIC_API_KEY が設定されていません。", file=sys.stderr)
        print("  export ANTHROPIC_API_KEY='your-api-key'", file=sys.stderr)
        sys.exit(1)

    result_text = ""
    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            model="claude-opus-4-6",
            system_prompt=SYSTEM_PROMPT,
            allowed_tools=["WebSearch", "WebFetch", "Bash", "Read"],
            cwd=str(BASE_DIR),
            thinking={"type": "adaptive"},
            permission_mode="bypassPermissions",
        ),
    ):
        if verbose:
            print(f"[メッセージ] {type(message).__name__}", flush=True)
        if isinstance(message, ResultMessage):
            result_text = message.result

    return result_text


async def run_interactive(verbose: bool = False):
    print("=" * 60)
    print("  S&P 500 分析エージェント（Claude Agent SDK版）")
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
        result = await run_agent(user_input, verbose=verbose)
        print("\n" + "=" * 60)
        print(result)
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="S&P 500 分析エージェント（Claude Agent SDK版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  export ANTHROPIC_API_KEY="your-key"
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
        anyio.run(run_interactive, args.verbose)
        return

    prompt = args.prompt or DEFAULT_PROMPT

    print(f"プロンプト: {prompt}\n分析中...\n")
    result = anyio.run(run_agent, prompt, args.verbose)
    print("\n" + "=" * 60)
    print("【分析結果】")
    print("=" * 60)
    print(result)
    print("=" * 60)


if __name__ == "__main__":
    main()
