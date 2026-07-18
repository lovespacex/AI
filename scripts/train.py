from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from minichat.config import ModelConfig, load_yaml
from minichat.data import PretrainBatchSource, SFTBatchSource
from minichat.model import MiniChatModel


def model_config_from_training(values: dict[str, Any]) -> ModelConfig:
    model_file = load_yaml(values["model_config"])
    return ModelConfig.from_dict(model_file.get("model", model_file))


def learning_rate(step: int, values: dict[str, Any]) -> float:
    maximum = float(values["learning_rate"])
    minimum = float(values.get("min_learning_rate", maximum * 0.1))
    warmup = int(values.get("warmup_steps", 0))
    total = int(values["max_steps"])
    if warmup and step < warmup:
        return maximum * (step + 1) / warmup
    ratio = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return minimum + 0.5 * (maximum - minimum) * (1.0 + math.cos(math.pi * ratio))


def save_checkpoint(
    output_dir: Path,
    step: int,
    model: MiniChatModel,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    train_config: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / f"step-{step:07d}.pt"
    temporary = output_dir / f".step-{step:07d}.tmp"
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": model_config.to_dict(),
        "train_config": train_config,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }
    torch.save(payload, temporary)
    os.replace(temporary, final_path)
    latest = output_dir / "latest.pt"
    latest_temp = output_dir / ".latest.pt"
    latest_temp.unlink(missing_ok=True)
    try:
        os.link(final_path, latest_temp)
    except OSError:
        # Filesystems without hard-link support still get a reliable latest checkpoint.
        import shutil

        shutil.copy2(final_path, latest_temp)
    os.replace(latest_temp, latest)
    checkpoints = sorted(output_dir.glob("step-*.pt"))
    for old in checkpoints[:-2]:
        old.unlink(missing_ok=True)
    return final_path


def evaluate(
    model: MiniChatModel,
    source: PretrainBatchSource | SFTBatchSource,
    batches: int,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(batches):
            inputs, labels = source.batch(batch_size, device)
            _, loss = model(inputs, labels)
            losses.append(float(loss))
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MiniChat from random initialization.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, help="Override max_steps for testing")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    values = load_yaml(args.config)
    if args.max_steps is not None:
        values["max_steps"] = args.max_steps
    seed = int(values.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the 200M configuration")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    model_config = model_config_from_training(values)
    mode = values["mode"]
    source_type = PretrainBatchSource if mode == "pretrain" else SFTBatchSource
    train_source = source_type(values["train_data"], model_config.max_seq_len, seed) if mode == "pretrain" else source_type(values["train_data"], seed)
    val_source = source_type(values["val_data"], model_config.max_seq_len, seed + 1) if mode == "pretrain" else source_type(values["val_data"], seed + 1)

    model = MiniChatModel(model_config).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model parameters: {parameters:,} ({parameters / 1e6:.2f}M)")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(values["learning_rate"]),
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=float(values.get("weight_decay", 0.1)),
        fused=True,
    )
    output_dir = Path(values["output_dir"])
    start_step = 0
    resume_path = output_dir / "latest.pt"
    should_resume = not args.no_resume and values.get("resume") == "auto" and resume_path.exists()
    if should_resume:
        checkpoint_values = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint_values["model"])
        optimizer.load_state_dict(checkpoint_values["optimizer"])
        start_step = int(checkpoint_values["step"])
        print(f"Resumed {resume_path} at step {start_step}")
    elif values.get("init_checkpoint"):
        init_path = Path(values["init_checkpoint"])
        if not init_path.exists():
            raise FileNotFoundError(f"Initial checkpoint does not exist: {init_path}")
        checkpoint_values = torch.load(init_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint_values["model"])
        print(f"Initialized model weights from {init_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "model_config.json").open("w", encoding="utf-8") as handle:
        json.dump(model_config.to_dict(), handle, ensure_ascii=False, indent=2)
    log_handle = (output_dir / "metrics.jsonl").open("a", encoding="utf-8", buffering=1)
    accumulation = int(values["gradient_accumulation_steps"])
    batch_size = int(values["micro_batch_size"])
    seq_len = model_config.max_seq_len
    model.train()
    optimizer.zero_grad(set_to_none=True)
    interval_start = time.perf_counter()
    interval_tokens = 0
    last_saved_step = start_step
    try:
        for step in range(start_step, int(values["max_steps"])):
            lr = learning_rate(step, values)
            for group in optimizer.param_groups:
                group["lr"] = lr
            running_loss = 0.0
            for _ in range(accumulation):
                inputs, labels = train_source.batch(batch_size, device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = model(inputs, labels)
                    scaled_loss = loss / accumulation
                scaled_loss.backward()
                running_loss += float(loss.detach())
                interval_tokens += batch_size * seq_len
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(values.get("grad_clip", 1.0))
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            completed_step = step + 1

            if completed_step % int(values["log_interval"]) == 0:
                elapsed = time.perf_counter() - interval_start
                metrics = {
                    "step": completed_step,
                    "loss": running_loss / accumulation,
                    "learning_rate": lr,
                    "grad_norm": float(grad_norm),
                    "tokens_per_second": interval_tokens / elapsed,
                    "max_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
                print(json.dumps(metrics, ensure_ascii=False))
                log_handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
                interval_start = time.perf_counter()
                interval_tokens = 0

            if completed_step % int(values["eval_interval"]) == 0:
                val_loss = evaluate(
                    model,
                    val_source,
                    int(values["eval_batches"]),
                    batch_size,
                    device,
                )
                metrics = {"step": completed_step, "validation_loss": val_loss}
                print(json.dumps(metrics, ensure_ascii=False))
                log_handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")

            if completed_step % int(values["save_interval"]) == 0:
                path = save_checkpoint(
                    output_dir, completed_step, model, optimizer, model_config, values
                )
                last_saved_step = completed_step
                print(f"Saved checkpoint: {path}")
        if (
            values.get("save_final", True)
            and int(values["max_steps"]) > start_step
            and last_saved_step != int(values["max_steps"])
        ):
            path = save_checkpoint(
                output_dir, int(values["max_steps"]), model, optimizer, model_config, values
            )
            print(f"Saved final checkpoint: {path}")
    finally:
        log_handle.close()


if __name__ == "__main__":
    main()
