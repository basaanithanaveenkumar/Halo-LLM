"""Token-level NLL: standard CE or focal (Lin et al.). Used by every variant loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def loss_settings(model) -> tuple[str, float, float]:
    train = getattr(getattr(model, "cfg", None), "train", None)
    if train is None:
        return "ce", 2.0, 1.0
    return train.loss_type, float(train.focal_gamma), float(train.focal_alpha)


def token_nll(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int | None = None,
    reduction: str = "mean",
    loss_type: str = "ce",
    focal_gamma: float = 2.0,
    focal_alpha: float = 1.0,
) -> torch.Tensor:
    """Per-token loss, same shape as ``targets`` when ``reduction='none'``.

    Focal: ``alpha * (1 - p_t)^gamma * CE``, with ``p_t = exp(-CE)``.
    """
    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
        ignore_index=-100 if ignore_index is None else ignore_index,
    ).reshape_as(targets)
    if loss_type == "focal":
        pt = torch.exp(-ce)
        nll = focal_alpha * (1.0 - pt).clamp(min=0.0).pow(focal_gamma) * ce
    elif loss_type == "ce":
        nll = ce
    else:
        raise ValueError(f"unknown loss_type {loss_type!r}; expected 'ce' or 'focal'")
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


def model_token_nll(model, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
    kind, gamma, alpha = loss_settings(model)
    kwargs.setdefault("loss_type", kind)
    kwargs.setdefault("focal_gamma", gamma)
    kwargs.setdefault("focal_alpha", alpha)
    return token_nll(logits, targets, **kwargs)
