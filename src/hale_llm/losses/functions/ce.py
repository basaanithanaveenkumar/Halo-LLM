from __future__ import annotations

import torch

from hale_llm.losses.reduce import per_token_cross_entropy
from hale_llm.losses.registry import register_token_loss


@register_token_loss("ce")
def cross_entropy_token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = -100,
    **_,
) -> torch.Tensor:
    return per_token_cross_entropy(logits, targets, ignore_index=ignore_index)
