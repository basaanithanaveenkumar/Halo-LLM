"""Local sliding-window mask (True = blocked), interleaved with global layers in `arch=lgt`."""

from __future__ import annotations

import torch


def sliding_window_mask(
    seq_len: int,
    window: int,
    device: torch.device | str,
    *,
    causal: bool,
) -> torch.Tensor:
    q = torch.arange(seq_len, device=device).unsqueeze(1)
    k = torch.arange(seq_len, device=device).unsqueeze(0)
    dist = q - k
    if causal:
        return dist > window
    return dist.abs() > window


def or_masks(*masks: torch.Tensor | None) -> torch.Tensor | None:
    out: torch.Tensor | None = None
    for mask in masks:
        if mask is None:
            continue
        out = mask if out is None else (out | mask)
    return out
