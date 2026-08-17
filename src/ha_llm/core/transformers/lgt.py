"""Local-global transformer stack (RMSNorm, GQA, dual RoPE, sliding-window + global)."""

from __future__ import annotations

import torch
import torch.nn as nn

from ha_llm.core.components import (
    RMSNorm,
    TransformerLayer,
    build_attention,
    build_ffn,
    or_masks,
    sliding_window_mask,
)


def is_global_layer(index: int, ratio: int) -> bool:
    return (index + 1) % (ratio + 1) == 0


class LGTStack(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float = 0.0,
        n_kv_heads: int | None = None,
        ffn_type: str = "geglu",
        moe_num_experts: int = 4,
        moe_top_k: int = 2,
        moe_num_shared: int = 1,
        use_time_cond: bool = False,
        sliding_window: int = 512,
        local_global_ratio: int = 5,
        rope_theta_local: float = 10_000.0,
        rope_theta_global: float = 1_000_000.0,
        p_rope: float = 0.25,
        causal_window: bool = True,
    ):
        super().__init__()
        if ffn_type == "mlp":
            ffn_type = "geglu"
        self.sliding_window = sliding_window
        self.causal_window = causal_window
        head_dim = d_model // n_heads
        layers = []
        flags: list[bool] = []
        for i in range(n_layers):
            global_layer = is_global_layer(i, local_global_ratio)
            flags.append(global_layer)
            if global_layer:
                kv = max(1, n_heads // 8)
                theta = rope_theta_global
                rotary_dim = max(2, int(p_rope * head_dim) - int(p_rope * head_dim) % 2)
            else:
                kv = n_kv_heads if n_kv_heads is not None else max(1, n_heads // 2)
                theta = rope_theta_local
                rotary_dim = head_dim
            layers.append(
                TransformerLayer(
                    d_model,
                    attention=build_attention(
                        "gqa",
                        d_model=d_model,
                        n_heads=n_heads,
                        dropout=dropout,
                        n_kv_heads=kv,
                        rope_theta=theta,
                        rotary_dim=rotary_dim,
                        qk_norm=True,
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
                    norm="rms",
                    post_norm=True,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.layer_is_global = flags
        self.ln_out = RMSNorm(d_model)

    def _mask(
        self,
        base: torch.Tensor | None,
        seq_len: int,
        device: torch.device,
        is_global: bool,
    ) -> torch.Tensor | None:
        if is_global:
            return base
        window = sliding_window_mask(
            seq_len, self.sliding_window, device, causal=self.causal_window
        )
        return or_masks(base, window)

    def forward(
        self,
        h: torch.Tensor,
        t_emb: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = h.size(1)
        device = h.device
        for layer, is_global in zip(self.layers, self.layer_is_global, strict=True):
            mask = self._mask(attn_mask, seq_len, device, is_global)
            h = layer(
                h,
                t_emb=t_emb,
                attn_mask=mask,
                key_padding_mask=key_padding_mask,
                positions=positions,
            )
        return self.ln_out(h)
