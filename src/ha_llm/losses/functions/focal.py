from __future__ import annotations

import torch

from ha_llm.losses.reduce import per_token_cross_entropy
from ha_llm.losses.registry import register_token_loss


@register_token_loss("focal")
def focal_token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = -100,
    focal_gamma: float = 2.0,
    focal_alpha: float = 1.0,
    **_,
) -> torch.Tensor:
    """Lin et al. focal: ``alpha * (1 - p_t)^gamma * CE``."""
    ce = per_token_cross_entropy(logits, targets, ignore_index=ignore_index)
    pt = torch.exp(-ce)
    return focal_alpha * (1.0 - pt).clamp(min=0.0).pow(focal_gamma) * ce
