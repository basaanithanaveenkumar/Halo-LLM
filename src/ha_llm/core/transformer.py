"""Shared transformer backbone assembled from registered attention + FFN blocks.

`attn_type` is the *mask* (causal / bidirectional / block_causal).
`attn_impl` is the *mechanism* (mha / gqa / mqa).
`ffn_type` is mlp or DeepSeek MoE.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ha_llm.core.blocks.attention import AttnType, build_attention, build_attn_mask
from ha_llm.core.blocks.embeddings import SinusoidalTimeEmbedding
from ha_llm.core.blocks.ffn import build_ffn
from ha_llm.core.blocks.layer import TransformerLayer

__all__ = [
    "AttnType",
    "SinusoidalTimeEmbedding",
    "TransformerLayer",
    "TransformerBackbone",
    "build_attn_mask",
]


class TransformerBackbone(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_length: int = 64,
        dropout: float = 0.0,
        attn_type: AttnType = "causal",
        use_time_cond: bool = False,
        block_size: int | None = None,
        tie_embeddings: bool = True,
        attn_impl: str = "mha",
        n_kv_heads: int | None = None,
        ffn_type: str = "mlp",
        moe_num_experts: int = 4,
        moe_top_k: int = 2,
        moe_num_shared: int = 1,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must divide n_heads={n_heads}")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_length = max_length
        self.attn_type = attn_type
        self.block_size = block_size
        self.use_time_cond = use_time_cond

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_length, d_model) * 0.02)
        self.time_embed = None
        if use_time_cond:
            self.time_embed = nn.Sequential(
                SinusoidalTimeEmbedding(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
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
                )
                for _ in range(n_layers)
            ]
        )
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.head.weight = self.token_emb.weight

    @classmethod
    def from_config(cls, vocab_size: int, model_cfg, **overrides):
        kwargs = {
            "vocab_size": vocab_size,
            "d_model": model_cfg.d_model,
            "n_heads": model_cfg.n_heads,
            "n_layers": model_cfg.n_layers,
            "d_ff": model_cfg.d_ff,
            "max_length": model_cfg.max_length,
            "dropout": model_cfg.dropout,
            "attn_type": model_cfg.attn_type,
            "use_time_cond": model_cfg.use_time_cond,
            "block_size": model_cfg.block_size,
            "attn_impl": model_cfg.attn_impl,
            "n_kv_heads": model_cfg.n_kv_heads,
            "ffn_type": model_cfg.ffn_type,
            "moe_num_experts": model_cfg.moe_num_experts,
            "moe_top_k": model_cfg.moe_top_k,
            "moe_num_shared": model_cfg.moe_num_shared,
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        pos_ids_len: int | None = None,
    ) -> torch.Tensor:
        b, s = token_ids.shape
        pos_len = pos_ids_len or min(s, self.max_length)
        if pos_len > self.max_length:
            raise ValueError(f"seq {pos_len} > max_length {self.max_length}")

        if s <= self.max_length:
            h = self.token_emb(token_ids) + self.pos_emb[:, :s, :]
        else:
            n = s // 2
            pe = self.pos_emb[:, :n, :]
            h = self.token_emb(token_ids) + torch.cat([pe, pe], dim=1)

        t_emb = None
        if self.use_time_cond and t is not None and self.time_embed is not None:
            t_emb = self.time_embed(t)

        kpm = (attention_mask == 0) if attention_mask is not None else None
        if attn_mask is None:
            stream_len = s // 2 if self.attn_type == "block_causal" else s
            attn_mask = build_attn_mask(
                self.attn_type,
                stream_len,
                token_ids.device,
                block_size=self.block_size,
            )

        for layer in self.layers:
            h = layer(h, t_emb=t_emb, attn_mask=attn_mask, key_padding_mask=kpm)
        return self.head(self.ln_out(h))
