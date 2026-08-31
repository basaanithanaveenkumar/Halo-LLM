"""Decorator-based plugin registries.

Import submodules for side effects when adding new plugins:
  - ``hale_llm.models`` registers variants (model/loss/sampler/collate)
  - ``hale_llm.core.optimizers`` registers optimizers
  - ``hale_llm.metrics.llm`` registers generic metrics

Two registry primitives:
  - ``NamedRegistry``: one name -> one object (models, losses, samplers, optimizers, collate, ...)
  - ``VariantRegistry``: one name -> many objects, resolved by training variant (metrics)
"""

from hale_llm.core.registry.base import NamedRegistry
from hale_llm.core.registry.plugins import (
    LOSSES,
    LOSSES_REGISTRY,
    METRICS,
    METRICS_REGISTRY,
    MODELS,
    MODELS_REGISTRY,
    OPTIMIZERS,
    OPTIMIZERS_REGISTRY,
    SAMPLERS,
    SAMPLERS_REGISTRY,
    VARIANTS,
    get_loss,
    get_model,
    get_optimizer,
    get_sampler,
    get_variant,
    instantiate_metrics,
    register_loss,
    register_metric,
    register_model,
    register_optimizer,
    register_sampler,
    register_variant,
)
from hale_llm.core.registry.variant import VariantRegistry

__all__ = [
    "NamedRegistry",
    "VariantRegistry",
    "MODELS_REGISTRY",
    "LOSSES_REGISTRY",
    "SAMPLERS_REGISTRY",
    "OPTIMIZERS_REGISTRY",
    "METRICS_REGISTRY",
    "MODELS",
    "VARIANTS",
    "LOSSES",
    "SAMPLERS",
    "OPTIMIZERS",
    "METRICS",
    "register_model",
    "register_variant",
    "register_loss",
    "register_sampler",
    "register_optimizer",
    "register_metric",
    "get_model",
    "get_variant",
    "get_loss",
    "get_sampler",
    "get_optimizer",
    "instantiate_metrics",
]
