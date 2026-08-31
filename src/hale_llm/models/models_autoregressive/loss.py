from __future__ import annotations

from hale_llm.core.registry import register_loss
from hale_llm.losses import model_token_nll


@register_loss("autoregressive")
def autoregressive_loss(model, batch: dict) -> torch.Tensor:
    logits = model(batch)
    return model_token_nll(
        model,
        logits[:, :-1, :],
        batch["labels"][:, :-1],
        ignore_index=-100,
        reduction="mean",
    )
