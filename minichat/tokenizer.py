from __future__ import annotations

import unicodedata
import json
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
]


def normalize_text(text: Any) -> str:
    if isinstance(text, (dict, list)):
        text = json.dumps(text, ensure_ascii=False, separators=(",", ":"))
    elif not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKC", text).replace("\x00", "")
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(lines).strip()


def build_tokenizer(vocab_size: int) -> tuple[Tokenizer, BpeTrainer]:
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>", byte_fallback=True))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=[*SPECIAL_TOKENS, "<|unk|>"],
        initial_alphabet=ByteLevel.alphabet(),
    )
    return tokenizer, trainer


def load_tokenizer(path: str | Path) -> Tokenizer:
    path = Path(path)
    if path.is_dir():
        path = path / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(path))
    for token in SPECIAL_TOKENS:
        if tokenizer.token_to_id(token) is None:
            raise ValueError(f"Tokenizer is missing required token {token}")
    return tokenizer
