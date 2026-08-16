from __future__ import annotations

# TODO: pluggable alpha(t) (log-linear / cosine) extracted from model.py
from abc import ABC, abstractmethod

import torch


class NoiseSchedule(ABC):
    @abstractmethod
    def alpha(self, t: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def alpha_prime(self, t: torch.Tensor) -> torch.Tensor: ...


class LogLinearSchedule(NoiseSchedule):
    def alpha(self, t):
        return 1.0 - t

    def alpha_prime(self, t):
        return -torch.ones_like(t)
