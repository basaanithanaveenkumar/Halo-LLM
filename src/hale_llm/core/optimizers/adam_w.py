from __future__ import annotations

from typing import Any

import torch

from hale_llm.core.optimizers.base import BaseOptimizer


class AdamWOptimizer(BaseOptimizer):
    def _make_optimizer(self, params: Any, config: dict[str, Any]):
        return torch.optim.AdamW(
            params,
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 0.0),
        )
