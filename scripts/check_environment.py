from __future__ import annotations

import json

import torch


def main() -> None:
    result = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        x.square().mean().backward()
        result.update(
            gpu=torch.cuda.get_device_name(),
            compute_capability=torch.cuda.get_device_capability(),
            bf16_supported=torch.cuda.is_bf16_supported(),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

