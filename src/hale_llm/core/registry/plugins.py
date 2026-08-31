"""Training/eval plugin registries.

NamedRegistry plugins (strict 1:1 name lookup):
  model, loss, sampler, optimizer

VariantRegistry plugins (name + variant resolution):
  metric
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from hale_llm.core.registry.base import NamedRegistry
from hale_llm.core.registry.variant import VariantRegistry

T = TypeVar("T")

MODELS_REGISTRY = NamedRegistry("model")
LOSSES_REGISTRY = NamedRegistry("loss")
SAMPLERS_REGISTRY = NamedRegistry("sampler")
OPTIMIZERS_REGISTRY = NamedRegistry("optimizer")
METRICS_REGISTRY = VariantRegistry("metric")

MODELS: dict[str, type] = MODELS_REGISTRY.items
VARIANTS = MODELS
LOSSES: dict[str, Callable[..., Any]] = LOSSES_REGISTRY.items
SAMPLERS: dict[str, Callable[..., Any]] = SAMPLERS_REGISTRY.items
OPTIMIZERS: dict[str, type] = OPTIMIZERS_REGISTRY.items
METRICS = METRICS_REGISTRY.items


def register_model(name: str) -> Callable[[type[T]], type[T]]:
    def deco(cls: type[T]) -> type[T]:
        MODELS_REGISTRY.add(name, cls)
        cls.variant_name = name  # type: ignore[attr-defined]
        return cls

    return deco


register_variant = register_model


def register_loss(name: str) -> Callable[[T], T]:
    def deco(fn: T) -> T:
        LOSSES_REGISTRY.add(name, fn)
        return fn

    return deco


def register_sampler(name: str) -> Callable[[T], T]:
    def deco(fn: T) -> T:
        SAMPLERS_REGISTRY.add(name, fn)
        return fn

    return deco


def register_optimizer(name: str) -> Callable[[type[T]], type[T]]:
    def deco(cls: type[T]) -> type[T]:
        OPTIMIZERS_REGISTRY.add(name.lower(), cls)
        return cls

    return deco


def register_metric(name: str, *, variants: tuple[str, ...] | None = None) -> Callable[[type[T]], type[T]]:
    def deco(cls: type[T]) -> type[T]:
        return METRICS_REGISTRY.add(name, cls, variants=variants)

    return deco


def get_model(name: str) -> type:
    return MODELS_REGISTRY.get(name)


get_variant = get_model


def get_loss(name: str) -> Callable[..., Any]:
    return LOSSES_REGISTRY.get(name)


def get_sampler(name: str) -> Callable[..., Any]:
    return SAMPLERS_REGISTRY.get(name)


def get_optimizer(name: str) -> type:
    return OPTIMIZERS_REGISTRY.get(name.lower())


def instantiate_metrics(names: list[str], variant: str) -> list[Any]:
    return METRICS_REGISTRY.instantiate(names, variant)
