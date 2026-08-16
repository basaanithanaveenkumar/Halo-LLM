"""Variant-agnostic metrics. Paradigm-specific metrics live in models_*/metrics.py."""

from __future__ import annotations

import math

import torch

from ha_llm.core.registry import register_metric
from ha_llm.metrics.base import Metric, MetricContext


class _MeanLossMixin:
    def __init__(self) -> None:
        self.total = 0.0
        self.n = 0

    def reset(self) -> None:
        self.total = 0.0
        self.n = 0

    def update(self, ctx: MetricContext) -> None:
        self.total += float(ctx.loss.item())
        self.n += 1

    def mean_loss(self) -> float:
        return self.total / max(self.n, 1)


@register_metric("loss")
class LossMetric(_MeanLossMixin, Metric):
    def compute(self) -> dict[str, float]:
        return {"loss": self.mean_loss()}


@register_metric("perplexity")
class PerplexityMetric(_MeanLossMixin, Metric):
    """exp(mean training loss). Comparable across variants only when the loss is token NLL."""

    def compute(self) -> dict[str, float]:
        mean = self.mean_loss()
        try:
            ppl = math.exp(mean)
        except OverflowError:
            ppl = float("inf")
        return {"perplexity": ppl}


@register_metric("bits_per_token")
class BitsPerTokenMetric(_MeanLossMixin, Metric):
    """mean_loss / ln(2). Same caveat as perplexity when the loss is not token NLL."""

    def compute(self) -> dict[str, float]:
        return {"bits_per_token": self.mean_loss() / math.log(2.0)}


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
        labels = ctx.batch.get("labels")
        if labels is None:
            return
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
        n = float(valid.float().sum().item())
        if n == 0:
            return
        self.correct += float(((pred == labels) & valid).float().sum().item())
        self.total += n

    def compute(self) -> dict[str, float]:
        return {"token_accuracy": self.correct / max(self.total, 1.0)}
