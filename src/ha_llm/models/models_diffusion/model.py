from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ha_llm.core.registry import register_loss, register_sampler, register_variant
from ha_llm.core.transformer import TransformerBackbone
from ha_llm.dataloader.collate import register_collate
from ha_llm.losses import model_token_nll


@register_collate("diffusion")
def collate_diffusion(input_ids: torch.Tensor, tokenizer, **_):
    pad_id = tokenizer.pad_token_id
    return {
        "input_ids": input_ids,
        "attention_mask": (input_ids != pad_id).long(),
        "pad_token_id": pad_id,
        "mask_token_id": tokenizer.mask_token_id,
    }


@register_variant("diffusion")
class MaskedDiffusionLM(nn.Module):
    def __init__(self, vocab_size: int, cfg):
        super().__init__()
        self.backbone = TransformerBackbone.from_config(vocab_size, cfg.model)
        self.vocab_size = vocab_size
        self.eps = cfg.sample.eps
        self.cfg = cfg

    def forward(self, batch: dict) -> torch.Tensor:
        return self.backbone(batch["x_t"], attention_mask=batch.get("attention_mask"), t=batch["t"])


def _corrupt(x0, t, mask_id):
    b, l = x0.shape
    mask_prob = t[:, None].expand(b, l)  # log-linear: P(mask)=t, alpha=1-t
    is_masked = torch.rand_like(x0, dtype=torch.float) < mask_prob
    x_t = torch.where(is_masked, torch.full_like(x0, mask_id), x0)
    return x_t, is_masked


@register_loss("diffusion")
def diffusion_loss(model, batch: dict) -> torch.Tensor:
    x0 = batch["input_ids"]
    b = x0.size(0)
    device = x0.device
    mask_id = batch["mask_token_id"]
    t = torch.rand(b, device=device) * (1.0 - model.eps) + model.eps
    x_t, is_masked = _corrupt(x0, t, mask_id)
    logits = model({**batch, "x_t": x_t, "t": t})
    loss_pos = is_masked
    if batch.get("attention_mask") is not None:
        loss_pos = loss_pos & batch["attention_mask"].bool()
    nll = model_token_nll(model, logits, x0, reduction="none")
    weight = (1.0 / t.clamp(min=model.eps))[:, None]
    return (nll * weight * loss_pos.float()).sum() / loss_pos.float().sum().clamp(min=1.0)


@register_sampler("diffusion")
@torch.no_grad()
def sample_diffusion(
    model,
    tokenizer,
    prompt_ids,
    max_new_tokens,
    temperature=1.0,
    sampling_steps=32,
    return_history=False,
    **_,
):
    model.eval()
    device = prompt_ids.device
    b, lp = prompt_ids.shape
    seq = lp + max_new_tokens
    mask_id = tokenizer.mask_token_id
    x = torch.full((b, seq), mask_id, device=device, dtype=torch.long)
    x[:, :lp] = prompt_ids
    history = [(1.0, x.clone())] if return_history else None
    t_steps = torch.linspace(1.0, model.eps, sampling_steps + 1, device=device)
    for i in range(sampling_steps):
        t_cur, t_next = t_steps[i], t_steps[i + 1]
        t = t_cur.expand(b)
        logits = model({"x_t": x, "t": t, "attention_mask": None}) / max(temperature, 1e-5)
        logits[..., mask_id] = -float("inf")
        pred = torch.multinomial(F.softmax(logits, dim=-1).reshape(-1, model.vocab_size), 1).reshape(b, seq)
        p_reveal = ((t_cur - t_next) / t_cur.clamp(min=1e-8)).clamp(0, 1)
        can = x.eq(mask_id)
        can[:, :lp] = False
        do = can & (torch.rand(b, seq, device=device) < p_reveal)
        x = torch.where(do, pred, x)
        if return_history:
            history.append((float(t_next), x.clone()))
    return (x, history) if return_history else x
