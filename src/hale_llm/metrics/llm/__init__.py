"""LLM eval metrics. Import this package so `@register_metric` runs."""

from hale_llm.metrics.llm.accuracy import TokenAccuracyMetric
from hale_llm.metrics.llm.bits import BitsPerTokenMetric
from hale_llm.metrics.llm.loss import LossMetric
from hale_llm.metrics.llm.perplexity import PerplexityMetric
from hale_llm.metrics.llm.topk import TopKAccuracyMetric

__all__ = [
    "LossMetric",
    "PerplexityMetric",
    "BitsPerTokenMetric",
    "TokenAccuracyMetric",
    "TopKAccuracyMetric",
]
