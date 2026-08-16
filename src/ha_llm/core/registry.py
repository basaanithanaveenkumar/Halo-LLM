"""Decorator-based registries. Train/CLI/eval resolve models by string name.

Never import a models_* package from core, training, data, or cli except via
`ha_llm.models` (which triggers registration). Model packages may import from core;
core never imports them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from loguru import logger

T = TypeVar("T")


class NamedRegistry:
    """Open/closed lookup table: add entries via `add`, never by editing callers."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.items: dict[str, Any] = {}

    def add(self, name: str, obj: Any) -> Any:
        if name in self.items:
            logger.error("duplicate {} registration {}", self.kind, name)
            raise ValueError(f"{self.kind} {name!r} already registered: {self.items[name]}")
        self.items[name] = obj
        logger.debug("registered {} {}", self.kind, name)
        return obj

    def get(self, name: str) -> Any:
        try:
            return self.items[name]
        except KeyError as e:
            logger.error("unknown {} {!r}; registered={}", self.kind, name, sorted(self.items))
            raise KeyError(
                f"unknown {self.kind} {name!r}; registered={sorted(self.items)}"
            ) from e

    def __contains__(self, name: str) -> bool:
        return name in self.items

    def __iter__(self):
        return iter(self.items)


class MetricRegistry:
    """A metric name may have a generic impl and/or variant-specific impls."""

    def __init__(self) -> None:
        self.items: dict[str, list[tuple[frozenset[str] | None, type]]] = {}

    def add(self, name: str, cls: type, variants: tuple[str, ...] | None = None) -> type:
        key = frozenset(variants) if variants is not None else None
        entries = self.items.setdefault(name, [])
        for existing, _ in entries:
            if existing == key:
                raise ValueError(f"metric {name!r} already registered for variants={variants}")
        entries.append((key, cls))
        cls.metric_name = name  # type: ignore[attr-defined]
        logger.debug("registered metric {} variants={}", name, variants)
        return cls

    def instantiate(self, names: list[str], variant: str) -> list[Any]:
        built: list[Any] = []
        for name in names:
            entries = self.items.get(name)
            if entries is None:
                logger.error("unknown metric {!r}; registered={}", name, sorted(self.items))
                raise KeyError(f"unknown metric {name!r}; registered={sorted(self.items)}")
            specific = [cls for variants, cls in entries if variants is not None and variant in variants]
            generic = [cls for variants, cls in entries if variants is None]
            cls = (specific or generic or [None])[0]
            if cls is None:
                logger.info("skipping metric {} (not defined for variant {})", name, variant)
                continue
            built.append(cls())
        return built


MODELS_REGISTRY = NamedRegistry("model")
LOSSES_REGISTRY = NamedRegistry("loss")
SAMPLERS_REGISTRY = NamedRegistry("sampler")
METRICS_REGISTRY = MetricRegistry()

MODELS: dict[str, type] = MODELS_REGISTRY.items
VARIANTS = MODELS
LOSSES: dict[str, Callable[..., Any]] = LOSSES_REGISTRY.items
SAMPLERS: dict[str, Callable[..., Any]] = SAMPLERS_REGISTRY.items
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


def get_model(name: str) -> type:
    return MODELS_REGISTRY.get(name)


get_variant = get_model


def get_loss(name: str) -> Callable[..., Any]:
    return LOSSES_REGISTRY.get(name)


def get_sampler(name: str) -> Callable[..., Any]:
    return SAMPLERS_REGISTRY.get(name)


def register_metric(name: str, *, variants: tuple[str, ...] | None = None) -> Callable[[type[T]], type[T]]:
    def deco(cls: type[T]) -> type[T]:
        return METRICS_REGISTRY.add(name, cls, variants=variants)

    return deco


def instantiate_metrics(names: list[str], variant: str) -> list[Any]:
    return METRICS_REGISTRY.instantiate(names, variant)
