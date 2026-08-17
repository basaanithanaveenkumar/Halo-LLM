import math

from ha_llm.core.registry import register_metric
from ha_llm.metrics.base import Metric
from ha_llm.metrics.llm.common import MeanLossMixin


@register_metric("perplexity")
class PerplexityMetric(MeanLossMixin, Metric):
    """exp(mean batch loss). Comparable across variants only when the loss is token NLL."""

    def compute(self) -> dict[str, float]:
        mean = self.mean_loss()
        try:
            ppl = math.exp(mean)
        except OverflowError:
            ppl = float("inf")
        return {"perplexity": ppl}
