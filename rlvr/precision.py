"""Execution precision shared by rollout generation and the GRPO forward.

Keep native CUDA bf16 on Ampere and newer. Pascal (and CPU) uses fp32;
there is deliberately no unscaled fp16 training fallback.
"""
from __future__ import annotations

import torch


def autocast_dtype(device: str | torch.device) -> torch.dtype | None:
    device = torch.device(device)
    # Query the requested ordinal, not the process's current CUDA device.
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8:
        return torch.bfloat16
    return None


def model_autocast(device: str | torch.device):
    device = torch.device(device)
    dtype = autocast_dtype(device)
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=dtype is not None)
