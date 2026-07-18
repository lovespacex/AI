from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class PretrainBatchSource:
    def __init__(self, path: str | Path, seq_len: int, seed: int = 42) -> None:
        self.path = Path(path)
        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.generator = np.random.default_rng(seed)
        if len(self.tokens) <= seq_len + 1:
            raise ValueError(f"Not enough tokens in {path}")

    def batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        starts = self.generator.integers(0, len(self.tokens) - self.seq_len - 1, size=batch_size)
        rows = np.stack(
            [np.asarray(self.tokens[start : start + self.seq_len + 1], dtype=np.int64) for start in starts]
        )
        values = torch.from_numpy(rows).to(device, non_blocking=True)
        return values[:, :-1], values[:, 1:]


class SFTBatchSource:
    def __init__(self, directory: str | Path, seed: int = 42) -> None:
        directory = Path(directory)
        with (directory / "meta.json").open("r", encoding="utf-8") as handle:
            self.meta = json.load(handle)
        self.seq_len = int(self.meta["seq_len"])
        self.records = int(self.meta["records"])
        width = self.seq_len + 1
        self.tokens = np.memmap(
            directory / "input_ids.bin", dtype=np.uint16, mode="r", shape=(self.records, width)
        )
        self.labels = np.memmap(
            directory / "labels.bin", dtype=np.int32, mode="r", shape=(self.records, width)
        )
        self.generator = np.random.default_rng(seed)

    def batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        indices = self.generator.integers(0, self.records, size=batch_size)
        tokens = torch.from_numpy(np.asarray(self.tokens[indices], dtype=np.int64)).to(device)
        labels = torch.from_numpy(np.asarray(self.labels[indices], dtype=np.int64)).to(device)
        return tokens[:, :-1], labels[:, 1:]

