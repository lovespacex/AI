from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, inv_freq: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(q.size(-2), device=q.device, dtype=inv_freq.dtype)
    angles = torch.outer(positions, inv_freq)
    angles = torch.repeat_interleave(angles, 2, dim=-1)
    cos = angles.cos()[None, None, :, :].to(q.dtype)
    sin = angles.sin()[None, None, :, :].to(q.dtype)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.dropout = config.dropout
        self.q_proj = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, self.inv_freq)
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
            enable_gqa=self.n_heads != self.n_kv_heads,
        )
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(output)


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.down_proj = nn.Linear(config.hidden_dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention = Attention(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.feed_forward(self.ffn_norm(x))


class MiniChatModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(DecoderLayer(config) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.dim, config.norm_eps)
        if not config.tie_embeddings:
            self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        else:
            self.output = None
        self.apply(self._init_weights)
        for layer in self.layers:
            nn.init.normal_(
                layer.attention.o_proj.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * config.n_layers),
            )
            nn.init.normal_(
                layer.feed_forward.down_proj.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * config.n_layers),
            )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {input_ids.size(1)} exceeds {self.config.max_seq_len}"
            )
        x = self.token_embeddings(input_ids)
        for layer in self.layers:
            if self.config.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        x = self.norm(x)
        weight = self.token_embeddings.weight if self.output is None else self.output.weight
        logits = F.linear(x, weight)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_id: int,
        temperature: float = 0.8,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        no_repeat_ngram_size: int = 3,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.max_seq_len :]
            logits, _ = self(context)
            next_logits = logits[:, -1, :].float()
            if repetition_penalty != 1.0:
                seen = torch.unique(context)
                values = next_logits[:, seen]
                next_logits[:, seen] = torch.where(
                    values < 0, values * repetition_penalty, values / repetition_penalty
                )
            if no_repeat_ngram_size > 1 and context.size(1) >= no_repeat_ngram_size:
                tokens = context[0].tolist()
                prefix_size = no_repeat_ngram_size - 1
                prefix = tuple(tokens[-prefix_size:])
                banned = {
                    tokens[index + prefix_size]
                    for index in range(len(tokens) - prefix_size)
                    if tuple(tokens[index : index + prefix_size]) == prefix
                }
                if banned:
                    next_logits[:, list(banned)] = float("-inf")
            if temperature <= 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative = sorted_probs.cumsum(dim=-1)
                remove = cumulative - sorted_probs > top_p
                sorted_probs.masked_fill_(remove, 0.0)
                sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True))
                sampled = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_indices.gather(-1, sampled)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            if bool(torch.all(next_token == eos_id)):
                break
        return input_ids
