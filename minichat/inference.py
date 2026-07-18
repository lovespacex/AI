from __future__ import annotations

import time
from pathlib import Path

import torch

from .config import ModelConfig
from .model import MiniChatModel
from .tokenizer import load_tokenizer


DEFAULT_SYSTEM = "你是一个友善、诚实、有帮助的中文助手。回答应当清楚、准确; 不知道时直接说明。"


def render_prompt(history: list[tuple[str, str]], user_text: str, system: str) -> str:
    parts = ["<|bos|><|system|>\n", system, "\n"]
    for question, answer in history:
        parts.extend(("<|user|>\n", question, "\n<|assistant|>\n", answer, "<|eos|>"))
    parts.extend(("<|user|>\n", user_text, "\n<|assistant|>\n"))
    return "".join(parts)


class ChatEngine:
    def __init__(
        self,
        checkpoint_path: str | Path,
        tokenizer_path: str | Path,
        system_prompt: str = DEFAULT_SYSTEM,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(Path(checkpoint_path), map_location=self.device, weights_only=False)
        self.config = ModelConfig.from_dict(checkpoint["model_config"])
        self.config.gradient_checkpointing = False
        self.model = MiniChatModel(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.eos_id = self.tokenizer.token_to_id("<|eos|>")
        self.system_prompt = system_prompt
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        self.checkpoint_path = str(Path(checkpoint_path).resolve())

    def _fit_history(
        self,
        history: list[tuple[str, str]],
        message: str,
        max_new_tokens: int,
    ) -> tuple[str, int]:
        budget = max(64, self.config.max_seq_len - max_new_tokens)
        retained = history[-8:]
        while True:
            prompt = render_prompt(retained, message, self.system_prompt)
            prompt_ids = self.tokenizer.encode(prompt).ids
            if len(prompt_ids) <= budget or not retained:
                return prompt, len(prompt_ids)
            retained.pop(0)

    def reply(
        self,
        message: str,
        history: list[tuple[str, str]],
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.85,
        repetition_penalty: float = 1.25,
    ) -> dict[str, object]:
        max_new_tokens = max(8, min(int(max_new_tokens), 256))
        temperature = max(0.0, min(float(temperature), 1.5))
        top_p = max(0.1, min(float(top_p), 1.0))
        repetition_penalty = max(1.0, min(float(repetition_penalty), 2.0))
        prompt, prompt_tokens = self._fit_history(history, message, max_new_tokens)
        prompt_ids = self.tokenizer.encode(prompt).ids
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        started = time.perf_counter()
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            generated = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_id=self.eos_id,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        answer_ids = generated[0, len(prompt_ids) :].tolist()
        if self.eos_id in answer_ids:
            answer_ids = answer_ids[: answer_ids.index(self.eos_id)]
        answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        if not answer:
            answer = "我暂时没有生成有效回答。"
        return {
            "answer": answer,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": len(answer_ids),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }

    def info(self) -> dict[str, object]:
        return {
            "name": "MiniChat 25M",
            "parameters": self.parameter_count,
            "context_length": self.config.max_seq_len,
            "device": str(self.device),
            "gpu": torch.cuda.get_device_name() if self.device.type == "cuda" else None,
        }

