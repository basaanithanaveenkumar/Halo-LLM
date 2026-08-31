"""Compatibility shim. Prefer `hale_llm.metrics.llm`."""

from hale_llm.metrics.llm import (
    BitsPerTokenMetric,
    LossMetric,
    PerplexityMetric,
    TokenAccuracyMetric,
    TopKAccuracyMetric,
)

__all__ = [
    "LossMetric",
    "PerplexityMetric",
    "BitsPerTokenMetric",
    "TokenAccuracyMetric",
    "TopKAccuracyMetric",
]
