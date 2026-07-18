from __future__ import annotations

import argparse
from minichat.inference import ChatEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with a trained MiniChat checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/sft-200m/latest.pt")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    args = parser.parse_args()

    engine = ChatEngine(args.checkpoint, args.tokenizer)
    history: list[tuple[str, str]] = []
    print("MiniChat 已加载。输入 /clear 清空上下文，/exit 退出。")
    while True:
        try:
            user_text = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text == "/exit":
            break
        if user_text == "/clear":
            history.clear()
            print("上下文已清空。")
            continue
        result = engine.reply(
            user_text,
            history,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        answer = str(result["answer"])
        print(f"MiniChat: {answer}")
        history.append((user_text, answer))


if __name__ == "__main__":
    main()
