"""Optimizer plugins. Import this package so `@register_optimizer` runs."""

from hale_llm.core.optimizers.adam import AdamOptimizer
from hale_llm.core.optimizers.adam_w import AdamWOptimizer
from hale_llm.core.optimizers.base import BaseOptimizer
from hale_llm.core.optimizers.muon import MuonOptimizer
from hale_llm.core.optimizers.sgd import SGDOptimizer
from hale_llm.core.registry import OPTIMIZERS, get_optimizer, register_optimizer

register_optimizer("adam")(AdamOptimizer)
register_optimizer("adamw")(AdamWOptimizer)
register_optimizer("sgd")(SGDOptimizer)
register_optimizer("muon")(MuonOptimizer)


def build_optimizer(name: str, params, config: dict):
    return get_optimizer(name)(params, config)


__all__ = [
    "BaseOptimizer",
    "AdamOptimizer",
    "AdamWOptimizer",
    "SGDOptimizer",
    "MuonOptimizer",
    "OPTIMIZERS",
    "register_optimizer",
    "get_optimizer",
    "build_optimizer",
]
