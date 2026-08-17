"""GPT-style pre-norm decoder stack (LayerNorm, learned positions live on the backbone)."""

from __future__ import annotations

import torch
import torch.nn as nn

from ha_llm.core.components import TransformerLayer, build_attention, build_ffn


class GPTStack(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float = 0.0,
        attn_impl: str = "mha",
        n_kv_heads: int | None = None,
        ffn_type: str = "mlp",
        moe_num_experts: int = 4,
        moe_top_k: int = 2,
        moe_num_shared: int = 1,
        use_time_cond: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_model,
                    attention=build_attention(
                        attn_impl,
                        d_model=d_model,
                        n_heads=n_heads,
                        dropout=dropout,
                        n_kv_heads=n_kv_heads,
                    ),
                    ffn=build_ffn(
                        ffn_type,
                        d_model=d_model,
                        d_ff=d_ff,
                        dropout=dropout,
                        moe_num_experts=moe_num_experts,
                        moe_top_k=moe_top_k,
                        moe_num_shared=moe_num_shared,
                    ),
                    use_time_cond=use_time_cond,
                    norm="layer",
                    post_norm=False,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln_out = nn.LayerNorm(d_model)

    def forward(
        self,
        h: torch.Tensor,
        t_emb: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            h = layer(
                h,
                t_emb=t_emb,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                positions=positions,
            )
        return self.ln_out(h)
