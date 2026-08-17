from __future__ import annotations

import torch

from ha_llm.core.registry import register_metric
from ha_llm.metrics.base import Metric, MetricContext


def _pred_and_labels(ctx: MetricContext):
    labels = ctx.batch.get("labels")
    if labels is None:
        return None, None, None
    logits = ctx.model(ctx.batch)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    pred = logits.argmax(dim=-1)
    pad_id = ctx.batch.get("pad_token_id")
    if ctx.variant == "mtp":
        pred = pred[:, :-1]
        labels = labels[:, 1:]
        valid = torch.ones_like(labels, dtype=torch.bool)
        if pad_id is not None:
            valid = labels.ne(pad_id)
    else:
        valid = labels.ne(-100)
        if pad_id is not None:
            valid = valid & labels.ne(pad_id)
    return pred, labels, valid


@register_metric("token_accuracy", variants=("autoregressive", "mtp"))
class TokenAccuracyMetric(Metric):
    """Greedy next-token accuracy on `labels`, ignoring -100 and pad."""

    def __init__(self) -> None:
        self.correct = 0.0
        self.total = 0.0

    def reset(self) -> None:
        self.correct = 0.0
        self.total = 0.0

    def update(self, ctx: MetricContext) -> None:
        pred, labels, valid = _pred_and_labels(ctx)
        if pred is None:
            return
        n = float(valid.float().sum().item())
        if n == 0:
            return
        self.correct += float(((pred == labels) & valid).float().sum().item())
        self.total += n

    def compute(self) -> dict[str, float]:
        return {"token_accuracy": self.correct / max(self.total, 1.0)}
