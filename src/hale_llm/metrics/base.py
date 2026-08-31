"""ABC for metrics. Implementations live in hale_llm.metrics.general and models_*/metrics.py."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class MetricContext:
    model: nn.Module
    batch: dict[str, Any]
    loss: torch.Tensor
    variant: str


class Metric(ABC):
    """Accumulate over an eval loader, then `compute()` a dict of scalar names."""

    metric_name: str = ""

    def reset(self) -> None:
        return

    @abstractmethod
    def update(self, ctx: MetricContext) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> dict[str, float]:
        raise NotImplementedError
