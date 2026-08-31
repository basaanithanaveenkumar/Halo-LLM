from __future__ import annotations

from typing import Any

import torch

from hale_llm.core.optimizers.base import BaseOptimizer


class MuonOptimizer(BaseOptimizer):
    def _make_optimizer(self, params: Any, config: dict[str, Any]):
        muon_cls = getattr(torch.optim, "Muon", None)
        if muon_cls is None:
            raise RuntimeError("torch.optim.Muon is not available in this PyTorch build")
        return muon_cls(
            params,
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 0.1),
            momentum=config.get("momentum", 0.95),
            nesterov=config.get("nesterov", True),
            ns_coefficients=config.get("ns_coefficients", (3.4445, -4.775, 2.0315)),
            eps=config.get("eps", 1e-07),
            ns_steps=config.get("ns_steps", 5),
            adjust_lr_fn=config.get("adjust_lr_fn", "original"),
        )
