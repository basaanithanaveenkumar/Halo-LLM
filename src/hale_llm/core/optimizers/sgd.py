from __future__ import annotations

from typing import Any

import torch

from hale_llm.core.optimizers.base import BaseOptimizer


class SGDOptimizer(BaseOptimizer):
    def _make_optimizer(self, params: Any, config: dict[str, Any]):
        return torch.optim.SGD(
            params,
            lr=config.get("lr", 1e-3),
            momentum=config.get("momentum", 0.9),
            weight_decay=config.get("weight_decay", 0.0),
            nesterov=config.get("nesterov", False),
        )
