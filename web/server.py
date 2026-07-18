from __future__ import annotations

import argparse
import json
import mimetypes
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from minichat.inference import ChatEngine


STATIC_DIR = Path(__file__).parent / "static"


class ChatRequestHandler(BaseHTTPRequestHandler):
    engine: ChatEngine
    inference_lock = threading.Lock()

    def log_message(self, format: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")

    def _send_json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_error(self, message: str, status: HTTPStatus) -> None:
        self._send_json({"error": message}, status)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            self._send_json({"ready": True, "model": self.engine.info()})
            return
        relative = "index.html" if path == "/" else path.lstrip("/")
        candidate = (STATIC_DIR / relative).resolve()
        try:
            candidate.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self._send_error("Not found", HTTPStatus.NOT_FOUND)
            return
        if not candidate.is_file():
            self._send_error("Not found", HTTPStatus.NOT_FOUND)
            return
        payload = candidate.read_bytes()
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/chat":
            self._send_error("Not found", HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                raise ValueError("请求大小无效")
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            message = data.get("message", "")
            if not isinstance(message, str) or not message.strip():
                raise ValueError("消息不能为空")
            if len(message) > 4_000:
                raise ValueError("消息过长")
            raw_history = data.get("history", [])
            if not isinstance(raw_history, list):
                raise ValueError("历史记录格式无效")
            history: list[tuple[str, str]] = []
            for item in raw_history[-8:]:
                if not isinstance(item, dict):
                    continue
                user = item.get("user")
                assistant = item.get("assistant")
                if isinstance(user, str) and isinstance(assistant, str):
                    history.append((user[:4_000], assistant[:8_000]))
            with self.inference_lock:
                result = self.engine.reply(
                    message.strip(),
                    history,
                    max_new_tokens=data.get("max_new_tokens", 128),
                    temperature=data.get("temperature", 0.7),
                    top_p=data.get("top_p", 0.85),
                    repetition_penalty=data.get("repetition_penalty", 1.25),
                )
            self._send_json(result)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._send_error(str(error), HTTPStatus.BAD_REQUEST)
        except Exception as error:
            print(f"Inference error: {error!r}")
            self._send_error("生成失败，请稍后重试", HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MiniChat local web interface.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint", default="checkpoints/quick-chat-25m/latest.pt")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    args = parser.parse_args()

    print("正在加载模型...")
    ChatRequestHandler.engine = ChatEngine(args.checkpoint, args.tokenizer)
    server = ThreadingHTTPServer((args.host, args.port), ChatRequestHandler)
    print(f"MiniChat Web 已启动: http://{args.host}:{args.port}")
    print(json.dumps(ChatRequestHandler.engine.info(), ensure_ascii=False))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

