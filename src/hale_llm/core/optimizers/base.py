from __future__ import annotations

from typing import Any


class BaseOptimizer:
    """Uniform (params, config) constructor over a torch.optim.Optimizer."""

    def __init__(self, params: Any, config: dict[str, Any]) -> None:
        self.params = params
        self.config = config
        self.optimizer = self._make_optimizer(params, config)

    def _make_optimizer(self, params: Any, config: dict[str, Any]):
        raise NotImplementedError

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.optimizer.step()

    def state_dict(self) -> dict[str, Any]:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self):
        return self.optimizer.param_groups
