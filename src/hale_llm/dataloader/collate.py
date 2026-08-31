from __future__ import annotations

from collections.abc import Callable
from typing import Any

from hale_llm.core.registry import NamedRegistry

COLLATE_REGISTRY = NamedRegistry("collate")
COLLATES: dict[str, Callable[..., dict[str, Any]]] = COLLATE_REGISTRY.items


def register_collate(name: str):
    def deco(fn):
        COLLATE_REGISTRY.add(name, fn)
        return fn

    return deco


def get_collate(name: str):
    return COLLATE_REGISTRY.get(name)
