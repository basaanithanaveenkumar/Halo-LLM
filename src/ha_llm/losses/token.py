"""Dispatch registered token losses. Variant task losses call `model_token_nll`."""

from __future__ import annotations

import torch

from ha_llm.losses import functions as _token_losses  # noqa: F401
from ha_llm.losses.reduce import reduce_per_token
from ha_llm.losses.registry import get_token_loss


def loss_settings(model) -> dict:
    train = getattr(getattr(model, "cfg", None), "train", None)
    if train is None:
        return {"loss_type": "ce", "focal_gamma": 2.0, "focal_alpha": 1.0, "label_smoothing": 0.1}
    return {
        "loss_type": train.loss_type,
        "focal_gamma": float(train.focal_gamma),
        "focal_alpha": float(train.focal_alpha),
        "label_smoothing": float(getattr(train, "label_smoothing", 0.1)),
    }


def token_nll(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int | None = None,
    reduction: str = "mean",
    loss_type: str = "ce",
    **kwargs,
) -> torch.Tensor:
    """Per-token loss via `losses.registry` (`ce`, `focal`, `label_smoothing`, `kl`, …)."""
    ignore = -100 if ignore_index is None else ignore_index
    nll = get_token_loss(loss_type)(logits, targets, ignore_index=ignore, **kwargs)
    return reduce_per_token(nll, targets, ignore_index=ignore_index, reduction=reduction)


def model_token_nll(model, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
    settings = loss_settings(model)
    kind = settings.pop("loss_type")
    kwargs.setdefault("loss_type", kind)
    for key, value in settings.items():
        kwargs.setdefault(key, value)
    return token_nll(logits, targets, **kwargs)
