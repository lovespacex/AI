from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 32_000
    dim: int = 896
    n_layers: int = 19
    n_heads: int = 14
    n_kv_heads: int = 7
    hidden_dim: int = 2_432
    max_seq_len: int = 1_024
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    gradient_checkpointing: bool = True
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.dim % self.n_heads:
            raise ValueError("dim must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.dim // self.n_heads) % 2:
            raise ValueError("attention head dimension must be even for RoPE")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return values

