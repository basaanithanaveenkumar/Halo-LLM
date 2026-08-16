"""Multi-token prediction — stub.

Adding this variant required only this folder + one import in variants/__init__.py
+ configs/mtp/small.yaml. Fill in the N-head loss and speculative-style decode.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ha_llm.core.registry import register_loss, register_sampler, register_variant
from ha_llm.core.transformer import TransformerBackbone
from ha_llm.dataloader.collate import register_collate
from ha_llm.losses import model_token_nll


@register_collate("mtp")
def collate_mtp(input_ids: torch.Tensor, tokenizer, **_):
    pad_id = tokenizer.pad_token_id
    return {
        "input_ids": input_ids,
        "attention_mask": (input_ids != pad_id).long(),
        "pad_token_id": pad_id,
        "labels": input_ids,
    }


@register_variant("mtp")
class MultiTokenPredictionLM(nn.Module):
    def __init__(self, vocab_size: int, cfg):
        super().__init__()
        self.n_heads = cfg.model.n_mtp_heads
        self.backbone = TransformerBackbone.from_config(
            vocab_size, cfg.model, tie_embeddings=False
        )
        self.extra_heads = nn.ModuleList(
            [nn.Linear(cfg.model.d_model, vocab_size, bias=False) for _ in range(self.n_heads - 1)]
        )
        self.vocab_size = vocab_size
        self.cfg = cfg

    def forward(self, batch: dict) -> list[torch.Tensor]:
        # TODO: return N logits tensors for tokens t+1 ... t+N from hidden states.
        logits_main = self.backbone(batch["input_ids"], attention_mask=batch.get("attention_mask"))
        return [logits_main]


@register_loss("mtp")
def mtp_loss(model, batch: dict) -> torch.Tensor:
    # TODO: sum CE over depth-k heads with shifted labels[:, k:].
    logits = model(batch)[0]
    labels = batch["labels"][:, 1:]
    return model_token_nll(
        model,
        logits[:, :-1, :],
        labels,
        ignore_index=batch["pad_token_id"],
        reduction="mean",
    )


@register_sampler("mtp")
@torch.no_grad()
def sample_mtp(model, tokenizer, prompt_ids, max_new_tokens, temperature=1.0, return_history=False, **_):
    # TODO: speculative / multi-token decode. Head-0 next-token for now.
    model.eval()
    x = prompt_ids
    pad_id = tokenizer.pad_token_id
    history = [(0.0, x.clone())] if return_history else None
    for step in range(max_new_tokens):
        if x.size(1) > model.backbone.max_length:
            x = x[:, -model.backbone.max_length :]
        out = model({"input_ids": x, "attention_mask": (x != pad_id).long()})
        logits = out[0][:, -1, :] / max(temperature, 1e-5)
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        x = torch.cat([x, nxt], dim=1)
        if return_history:
            history.append(((step + 1) / max_new_tokens, x.clone()))
    return (x, history) if return_history else x
