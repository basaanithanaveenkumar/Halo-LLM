from __future__ import annotations

import torch

from hale_llm.core.registry import register_metric
from hale_llm.metrics.base import Metric, MetricContext


@register_metric("masked_accuracy", variants=("flow_matching",))
class FlowMatchingMaskedAccuracy(Metric):
    def __init__(self) -> None:
        self.correct = 0.0
        self.total = 0.0

    def reset(self) -> None:
        self.correct = 0.0
        self.total = 0.0

    def update(self, ctx: MetricContext) -> None:
        x1 = ctx.batch["input_ids"]
        b, l = x1.shape
        t = torch.rand(b, device=x1.device) * (1.0 - ctx.model.eps)
        keep_mask = (1.0 - t).clamp(0, 1)[:, None].expand(b, l)
        is_masked = torch.rand(b, l, device=x1.device) < keep_mask
        x_t = torch.where(is_masked, torch.full_like(x1, ctx.batch["mask_token_id"]), x1)
        logits = ctx.model({**ctx.batch, "x_t": x_t, "t": t})
        valid = is_masked
        if ctx.batch.get("attention_mask") is not None:
            valid = valid & ctx.batch["attention_mask"].bool()
        pred = logits.argmax(dim=-1)
        n = float(valid.float().sum().item())
        if n == 0:
            return
        self.correct += float(((pred == x1) & valid).float().sum().item())
        self.total += n

    def compute(self) -> dict[str, float]:
        return {"masked_accuracy": self.correct / max(self.total, 1.0)}
