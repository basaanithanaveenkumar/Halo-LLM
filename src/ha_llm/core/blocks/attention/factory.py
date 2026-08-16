from __future__ import annotations

from ha_llm.core.blocks.attention.grouped import GroupedQueryAttention
from ha_llm.core.blocks.attention.mha import TorchMultiHeadAttention
from ha_llm.core.registry import NamedRegistry

ATTENTION = NamedRegistry("attention")


def register_attention(name: str):
    def deco(cls):
        ATTENTION.add(name, cls)
        return cls

    return deco


register_attention("mha")(TorchMultiHeadAttention)
register_attention("gqa")(GroupedQueryAttention)
register_attention("mqa")(GroupedQueryAttention)


def build_attention(
    impl: str,
    *,
    d_model: int,
    n_heads: int,
    dropout: float = 0.0,
    n_kv_heads: int | None = None,
) -> TorchMultiHeadAttention | GroupedQueryAttention:
    if impl == "mha":
        return ATTENTION.get("mha")(d_model, n_heads, dropout)
    if impl == "mqa":
        return ATTENTION.get("mqa")(d_model, n_heads, n_kv_heads=1, dropout=dropout)
    if impl == "gqa":
        kv = n_kv_heads if n_kv_heads is not None else max(1, n_heads // 4)
        return ATTENTION.get("gqa")(d_model, n_heads, n_kv_heads=kv, dropout=dropout)
    raise KeyError(f"unknown attention impl {impl!r}; registered={sorted(ATTENTION.items)}")
