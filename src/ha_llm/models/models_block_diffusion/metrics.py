from __future__ import annotations

import torch

from ha_llm.core.registry import register_metric
from ha_llm.metrics.base import Metric, MetricContext


@register_metric("masked_accuracy", variants=("block_diffusion",))
class BlockDiffusionMaskedAccuracy(Metric):
    def __init__(self) -> None:
        self.correct = 0.0
        self.total = 0.0

    def reset(self) -> None:
        self.correct = 0.0
        self.total = 0.0

    def update(self, ctx: MetricContext) -> None:
        x = ctx.batch["input_ids"]
        b, n = x.shape
        bs = ctx.model.block_size
        num_blocks = n // bs
        t_block = torch.rand(b, num_blocks, device=x.device) * (1.0 - ctx.model.eps) + ctx.model.eps
        t_noised = t_block.repeat_interleave(bs, dim=1)
        is_masked = torch.rand(b, n, device=x.device) < t_noised
        x_t = torch.where(is_masked, torch.full_like(x, ctx.batch["mask_token_id"]), x)
        logits = ctx.model({**ctx.batch, "x_clean": x, "x_noised": x_t, "t_noised": t_noised})
        valid = is_masked
        if ctx.batch.get("attention_mask") is not None:
            valid = valid & ctx.batch["attention_mask"].bool()
        pred = logits.argmax(dim=-1)
        n = float(valid.float().sum().item())
        if n == 0:
            return
        self.correct += float(((pred == x) & valid).float().sum().item())
        self.total += n

    def compute(self) -> dict[str, float]:
        return {"masked_accuracy": self.correct / max(self.total, 1.0)}
