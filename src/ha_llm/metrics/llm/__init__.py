"""LLM eval metrics. Import this package so `@register_metric` runs."""

from ha_llm.metrics.llm.accuracy import TokenAccuracyMetric
from ha_llm.metrics.llm.bits import BitsPerTokenMetric
from ha_llm.metrics.llm.loss import LossMetric
from ha_llm.metrics.llm.perplexity import PerplexityMetric
from ha_llm.metrics.llm.topk import TopKAccuracyMetric

__all__ = [
    "LossMetric",
    "PerplexityMetric",
    "BitsPerTokenMetric",
    "TokenAccuracyMetric",
    "TopKAccuracyMetric",
]
