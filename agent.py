#!/usr/bin/env python3
"""
S&P 500 分析エージェント
Claude Agent SDK を使用した S&P 500 予測・分析の専門エージェント

使い方:
    python agent.py
    python agent.py --prompt "明日のS&P500を予測してください"
    python agent.py --interactive
"""

import anyio
import argparse
import sys
from pathlib import Path

from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage, SystemMessage

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
1. 最新の市場データと経済指標をWebで調査してから分析を行う
2. 複数の視点（テクニカル・ファンダメンタルズ・センチメント）を統合して総合判断する
3. 予測には必ず根拠と信頼度（低/中/高）を明示する
4. リスク要因（下落シナリオ）も必ず提示する
5. 既存のLSTM予測モデル（predict.py）の結果も参照する
6. 日本語で回答する（専門用語は英語のまま可）
7. 数値・パーセント・日付は具体的に示す

【免責事項の付記】
分析結果は情報提供目的であり、投資助言ではありません。
投資判断はご自身の責任で行ってください。"""


async def run_agent(prompt: str, verbose: bool = False) -> str:
    """S&P 500 分析エージェントを実行"""
    result_text = ""

    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            cwd=str(BASE_DIR),
            model="claude-opus-4-6",
            system_prompt=SYSTEM_PROMPT,
            allowed_tools=["WebSearch", "WebFetch", "Bash", "Read", "Glob"],
            permission_mode="acceptEdits",
            max_turns=20,
            thinking={"type": "adaptive"},
        ),
    ):
        if isinstance(message, SystemMessage):
            if verbose and message.subtype == "init":
                session_id = message.data.get("session_id", "")
                print(f"[セッション開始] ID: {session_id}", flush=True)
        elif isinstance(message, ResultMessage):
            result_text = message.result
            if verbose:
                print(f"\n[完了] stop_reason: {message.stop_reason}", flush=True)

    return result_text


async def run_interactive():
    """対話型モードでエージェントを実行"""
    print("=" * 60)
    print("  S&P 500 分析エージェント (対話モード)")
    print("  終了するには 'exit' または 'quit' と入力してください")
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
        result = await run_agent(user_input, verbose=True)
        print("\n" + "=" * 60)
        print(result)
        print("=" * 60)


async def main_async(args):
    if args.interactive:
        await run_interactive()
        return

    prompt = args.prompt or (
        "現在のS&P 500の状況を分析し、今後1週間の値動きを予測してください。"
        "最新の市場データ、VIX、主要経済指標を考慮した上で、"
        "強気・弱気それぞれのシナリオと重要な価格水準を教えてください。"
    )

    print(f"プロンプト: {prompt}\n")
    print("分析中...\n")

    result = await run_agent(prompt, verbose=args.verbose)

    print("\n" + "=" * 60)
    print("【分析結果】")
    print("=" * 60)
    print(result)
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="S&P 500 分析エージェント",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python agent.py
  python agent.py --prompt "今週のS&P500の主要サポートとレジスタンスは？"
  python agent.py --interactive
  python agent.py --verbose
        """,
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="エージェントへの質問・指示（省略時はデフォルト分析を実行）",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="対話モードで起動",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="詳細ログを表示",
    )
    args = parser.parse_args()

    try:
        anyio.run(main_async, args)
    except KeyboardInterrupt:
        print("\n中断されました。")
        sys.exit(0)


if __name__ == "__main__":
    main()
