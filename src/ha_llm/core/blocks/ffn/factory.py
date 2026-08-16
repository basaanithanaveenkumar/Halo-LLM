from __future__ import annotations

from ha_llm.core.blocks.ffn.mlp import DenseMLP
from ha_llm.core.blocks.ffn.moe import DeepseekMoE
from ha_llm.core.registry import NamedRegistry

FFN = NamedRegistry("ffn")


def register_ffn(name: str):
    def deco(cls):
        FFN.add(name, cls)
        return cls

    return deco


register_ffn("mlp")(DenseMLP)
register_ffn("moe")(DeepseekMoE)


def build_ffn(
    kind: str,
    *,
    d_model: int,
    d_ff: int,
    dropout: float = 0.0,
    moe_num_experts: int = 4,
    moe_top_k: int = 2,
    moe_num_shared: int = 1,
):
    if kind == "mlp":
        return FFN.get("mlp")(d_model, d_ff, dropout)
    if kind == "moe":
        return FFN.get("moe")(
            d_model,
            d_ff,
            n_experts=moe_num_experts,
            top_k=moe_top_k,
            n_shared=moe_num_shared,
            dropout=dropout,
        )
    raise KeyError(f"unknown ffn {kind!r}; registered={sorted(FFN.items)}")
