from __future__ import annotations

import torch
import torch.nn.functional as F

from hale_llm.core.registry import register_sampler


@register_sampler("autoregressive")
@torch.no_grad()
def sample_autoregressive(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    return_history: bool = False,
    **_,
):
    model.eval()
    x = prompt_ids
    pad_id = tokenizer.pad_token_id
    history = [(0.0, x.clone())] if return_history else None
    for step in range(max_new_tokens):
        if x.size(1) > model.backbone.max_length:
            x = x[:, -model.backbone.max_length :]
        attn = (x != pad_id).long()
        logits = model({"input_ids": x, "attention_mask": attn})
        next_logits = logits[:, -1, :] / max(temperature, 1e-5)
        nxt = torch.multinomial(F.softmax(next_logits, dim=-1), 1)
        x = torch.cat([x, nxt], dim=1)
        if return_history:
            history.append(((step + 1) / max_new_tokens, x.clone()))
    return (x, history) if return_history else x
