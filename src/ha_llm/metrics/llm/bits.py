import math

from ha_llm.core.registry import register_metric
from ha_llm.metrics.base import Metric
from ha_llm.metrics.llm.common import MeanLossMixin


@register_metric("bits_per_token")
class BitsPerTokenMetric(MeanLossMixin, Metric):
    """mean_loss / ln(2). Same caveat as perplexity when the loss is not token NLL."""

    def compute(self) -> dict[str, float]:
        return {"bits_per_token": self.mean_loss() / math.log(2.0)}
