"""Per-token loss reduction (ignore_index + mean/sum/none)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def per_token_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    ).reshape_as(targets)


def reduce_per_token(
    nll: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int | None,
    reduction: str,
) -> torch.Tensor:
    if reduction == "none":
        return nll
    if ignore_index is None:
        if reduction == "sum":
            return nll.sum()
        return nll.mean()
    valid = targets != ignore_index
    nll = nll * valid.float()
    if reduction == "sum":
        return nll.sum()
    return nll.sum() / valid.float().sum().clamp(min=1.0)
