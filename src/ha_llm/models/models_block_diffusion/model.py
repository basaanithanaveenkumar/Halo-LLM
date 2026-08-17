from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ha_llm.core.backbones import build_backbone
from ha_llm.core.registry import register_loss, register_sampler, register_variant
from ha_llm.dataloader.collate import register_collate
from ha_llm.losses import model_token_nll


@register_collate("block_diffusion")
def collate_block_diffusion(input_ids: torch.Tensor, tokenizer, **_):
    pad_id = tokenizer.pad_token_id
    return {
        "input_ids": input_ids,
        "attention_mask": (input_ids != pad_id).long(),
        "pad_token_id": pad_id,
        "mask_token_id": tokenizer.mask_token_id,
    }


@register_variant("block_diffusion")
class BlockDiffusionLM(nn.Module):
    def __init__(self, vocab_size: int, cfg):
        super().__init__()
        block_size = cfg.model.block_size or 16
        self.backbone = build_backbone(
            vocab_size, cfg.model, block_size=block_size
        )
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.eps = cfg.sample.eps
        self.steps_per_block = cfg.sample.sampling_steps
        self.cfg = cfg

    def forward(self, batch: dict) -> torch.Tensor:
        x_clean, x_noised = batch["x_clean"], batch["x_noised"]
        t_noised = batch["t_noised"]
        t_clean = torch.zeros_like(t_noised)
        tokens = torch.cat([x_clean, x_noised], dim=1)
        t = torch.cat([t_clean, t_noised], dim=1)
        attn = batch.get("attention_mask")
        if attn is not None:
            attn = torch.cat([attn, attn], dim=1)
        logits = self.backbone(tokens, attention_mask=attn, t=t)
        n = x_clean.size(1)
        return logits[:, n:, :]


@register_loss("block_diffusion")
def block_diffusion_loss(model, batch: dict) -> torch.Tensor:
    x = batch["input_ids"]
    b, n = x.shape
    bs = model.block_size
    if n % bs != 0:
        raise ValueError(f"seq_len {n} must be divisible by block_size {bs}")
    num_blocks = n // bs
    t_block = torch.rand(b, num_blocks, device=x.device) * (1.0 - model.eps) + model.eps
    t_noised = t_block.repeat_interleave(bs, dim=1)
    is_masked = torch.rand(b, n, device=x.device) < t_noised
    x_t = torch.where(is_masked, torch.full_like(x, batch["mask_token_id"]), x)
    logits = model({**batch, "x_clean": x, "x_noised": x_t, "t_noised": t_noised})
    loss_pos = is_masked
    if batch.get("attention_mask") is not None:
        loss_pos = loss_pos & batch["attention_mask"].bool()
    nll = model_token_nll(model, logits, x, reduction="none")
    weight = 1.0 / t_noised.clamp(min=model.eps)
    return (nll * weight * loss_pos.float()).sum() / loss_pos.float().sum().clamp(min=1.0)


@register_sampler("block_diffusion")
@torch.no_grad()
def sample_block_diffusion(
    model,
    tokenizer,
    prompt_ids,
    max_new_tokens,
    temperature=1.0,
    sampling_steps=None,
    return_history=False,
    **_,
):
    """AR across blocks, absorbing-state diffusion within the current block."""
    model.eval()
    device = prompt_ids.device
    n = model.backbone.max_length
    block_size = model.block_size
    if n % block_size != 0:
        raise ValueError(f"max_length {n} must be divisible by block_size {block_size}")
    num_blocks = n // block_size
    steps = sampling_steps or model.steps_per_block
    bsz = prompt_ids.size(0)
    mask_id = tokenizer.mask_token_id

    x = torch.full((bsz, n), mask_id, dtype=torch.long, device=device)
    lp = min(prompt_ids.size(1), n)
    x[:, :lp] = prompt_ids[:, :lp]
    is_prompt = torch.zeros(bsz, n, dtype=torch.bool, device=device)
    is_prompt[:, :lp] = True
    attention_mask = torch.ones(bsz, n, dtype=torch.long, device=device)

    meta0 = {"block": -1, "num_blocks": num_blocks, "block_size": block_size}
    history = [(1.0, x.clone(), meta0)] if return_history else None

    for b_idx in range(num_blocks):
        start, end = b_idx * block_size, (b_idx + 1) * block_size
        if is_prompt[:, start:end].all():
            continue
        t_steps = torch.linspace(1.0, 0.0, steps + 1, device=device)
        for step in range(steps):
            t_cur, t_next = t_steps[step].item(), t_steps[step + 1].item()
            t_noised = torch.zeros(bsz, n, device=device)
            t_noised[:, start:end] = t_cur
            logits = model(
                {
                    "x_clean": x,
                    "x_noised": x,
                    "t_noised": t_noised,
                    "attention_mask": attention_mask,
                }
            )
            logits[:, start:end, mask_id] = -float("inf")
            probs = F.softmax(
                logits[:, start:end, :] / max(temperature, 1e-5), dim=-1
            )
            sampled = torch.multinomial(
                probs.reshape(-1, model.vocab_size), num_samples=1
            ).reshape(bsz, block_size)
            block_tokens = x[:, start:end]
            is_masked = block_tokens.eq(mask_id)
            p_unmask = 1.0 if t_cur <= 1e-8 else (t_cur - t_next) / t_cur
            do_unmask = (
                is_masked
                & (torch.rand(bsz, block_size, device=device) < p_unmask)
                & ~is_prompt[:, start:end]
            )
            x[:, start:end] = torch.where(do_unmask, sampled, block_tokens)
            if return_history:
                history.append(
                    (
                        t_cur,
                        x.clone(),
                        {
                            "block": b_idx,
                            "num_blocks": num_blocks,
                            "block_size": block_size,
                        },
                    )
                )

        still = x[:, start:end].eq(mask_id) & ~is_prompt[:, start:end]
        if still.any():
            t_noised = torch.zeros(bsz, n, device=device)
            logits = model(
                {
                    "x_clean": x,
                    "x_noised": x,
                    "t_noised": t_noised,
                    "attention_mask": attention_mask,
                }
            )
            logits[:, start:end, mask_id] = -float("inf")
            final_pred = logits[:, start:end, :].argmax(dim=-1)
            x[:, start:end] = torch.where(still, final_pred, x[:, start:end])
            if return_history:
                history.append(
                    (
                        0.0,
                        x.clone(),
                        {
                            "block": b_idx,
                            "num_blocks": num_blocks,
                            "block_size": block_size,
                        },
                    )
                )

    return (x, history) if return_history else x
