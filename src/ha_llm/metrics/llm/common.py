from ha_llm.metrics.base import Metric, MetricContext


class MeanLossMixin:
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
