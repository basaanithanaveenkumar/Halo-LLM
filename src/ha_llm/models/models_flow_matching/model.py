from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ha_llm.core.backbones import build_backbone
from ha_llm.core.registry import register_loss, register_sampler, register_variant
from ha_llm.dataloader.collate import register_collate
from ha_llm.losses import model_token_nll


@register_collate("flow_matching")
def collate_flow_matching(input_ids: torch.Tensor, tokenizer, **_):
    pad_id = tokenizer.pad_token_id
    return {
        "input_ids": input_ids,  # x1 clean
        "attention_mask": (input_ids != pad_id).long(),
        "pad_token_id": pad_id,
        "mask_token_id": tokenizer.mask_token_id,
    }


@register_variant("flow_matching")
class FlowMatchingLM(nn.Module):
    def __init__(self, vocab_size: int, cfg):
        super().__init__()
        self.backbone = build_backbone(vocab_size, cfg.model)
        self.vocab_size = vocab_size
        self.eps = cfg.sample.eps
        self.cfg = cfg

    def forward(self, batch: dict) -> torch.Tensor:
        return self.backbone(batch["x_t"], attention_mask=batch.get("attention_mask"), t=batch["t"])


@register_loss("flow_matching")
def flow_matching_loss(model, batch: dict) -> torch.Tensor:
    x1 = batch["input_ids"]
    b, l = x1.shape
    device = x1.device
    t = torch.rand(b, device=device) * (1.0 - model.eps)
    keep_mask = (1.0 - t).clamp(0, 1)[:, None].expand(b, l)
    is_masked = torch.rand(b, l, device=device) < keep_mask
    x_t = torch.where(is_masked, torch.full_like(x1, batch["mask_token_id"]), x1)
    logits = model({**batch, "x_t": x_t, "t": t})
    loss_pos = is_masked
    if batch.get("attention_mask") is not None:
        loss_pos = loss_pos & batch["attention_mask"].bool()
    nll = model_token_nll(model, logits, x1, reduction="none")
    weight = (1.0 / (1.0 - t).clamp(min=model.eps))[:, None]
    return (nll * weight * loss_pos.float()).sum() / loss_pos.float().sum().clamp(min=1.0)


@register_sampler("flow_matching")
@torch.no_grad()
def sample_flow_matching(
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
    dt = 1.0 / sampling_steps
    gen = torch.arange(seq, device=device).ge(lp).unsqueeze(0)
    history = [(0.0, x.clone())] if return_history else None
    for step in range(sampling_steps):
        t_val = step * dt
        t = torch.full((b,), t_val, device=device)
        logits = model({"x_t": x, "t": t}) / max(temperature, 1e-5)
        logits[..., mask_id] = -float("inf")
        pred = torch.multinomial(F.softmax(logits, dim=-1).reshape(-1, model.vocab_size), 1).reshape(b, seq)
        p_unmask = min(dt / max(1.0 - t_val, model.eps), 1.0)
        do = x.eq(mask_id) & gen & (torch.rand(b, seq, device=device) < p_unmask)
        x = torch.where(do, pred, x)
        if return_history:
            history.append((t_val + dt, x.clone()))
    return (x, history) if return_history else x
