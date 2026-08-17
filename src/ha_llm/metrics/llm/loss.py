from ha_llm.core.registry import register_metric
from ha_llm.metrics.llm.common import MeanLossMixin
from ha_llm.metrics.base import Metric


@register_metric("loss")
class LossMetric(MeanLossMixin, Metric):
    def compute(self) -> dict[str, float]:
        return {"loss": self.mean_loss()}
