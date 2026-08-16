"""Apply PyTorch MHA-style masks (True = blocked) onto attention scores."""

from __future__ import annotations

import torch


def apply_attn_masks(
    scores: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    key_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask
    if key_padding_mask is not None:
        scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
    return scores
