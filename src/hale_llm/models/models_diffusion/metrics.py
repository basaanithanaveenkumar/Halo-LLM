from __future__ import annotations

import torch

from hale_llm.core.registry import register_metric
from hale_llm.metrics.base import Metric, MetricContext
from hale_llm.models.models_diffusion.model import _corrupt


@register_metric("masked_accuracy", variants=("diffusion",))
class DiffusionMaskedAccuracy(Metric):
    def __init__(self) -> None:
        self.correct = 0.0
        self.total = 0.0

    def reset(self) -> None:
        self.correct = 0.0
        self.total = 0.0

    def update(self, ctx: MetricContext) -> None:
        x0 = ctx.batch["input_ids"]
        b = x0.size(0)
        t = torch.rand(b, device=x0.device) * (1.0 - ctx.model.eps) + ctx.model.eps
        x_t, is_masked = _corrupt(x0, t, ctx.batch["mask_token_id"])
        logits = ctx.model({**ctx.batch, "x_t": x_t, "t": t})
        valid = is_masked
        if ctx.batch.get("attention_mask") is not None:
            valid = valid & ctx.batch["attention_mask"].bool()
        pred = logits.argmax(dim=-1)
        n = float(valid.float().sum().item())
        if n == 0:
            return
        self.correct += float(((pred == x0) & valid).float().sum().item())
        self.total += n

    def compute(self) -> dict[str, float]:
        return {"masked_accuracy": self.correct / max(self.total, 1.0)}
