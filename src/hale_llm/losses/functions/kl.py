from __future__ import annotations

import torch
import torch.nn.functional as F

from hale_llm.losses.registry import register_token_loss


@register_token_loss("kl")
def kl_token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    ignore_index: int = -100,
    **_,
) -> torch.Tensor:
    """Token-wise KL(teacher || student). `targets` only provide the ignore mask."""
    temp = max(float(temperature), 1e-5)
    log_p = F.log_softmax(logits.float() / temp, dim=-1)
    q = F.softmax(teacher_logits.float() / temp, dim=-1)
    vocab = logits.size(-1)
    kl = F.kl_div(log_p.reshape(-1, vocab), q.reshape(-1, vocab), reduction="none").sum(-1)
    return (kl * (temp * temp)).reshape_as(targets)
