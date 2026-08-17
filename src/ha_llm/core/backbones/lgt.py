"""Local-global token LM: RoPE (no learned pos) + LGTStack + tied LM head."""

from __future__ import annotations

import torch.nn as nn

from ha_llm.core.backbones.sequence import SequenceBackbone
from ha_llm.core.components.attention import AttnType
from ha_llm.core.transformers.lgt import LGTStack


class LGTBackbone(SequenceBackbone):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_length: int = 64,
        dropout: float = 0.0,
        attn_type: AttnType = "bidirectional",
        use_time_cond: bool = False,
        block_size: int | None = None,
        tie_embeddings: bool = True,
        n_kv_heads: int | None = None,
        ffn_type: str = "geglu",
        moe_num_experts: int = 4,
        moe_top_k: int = 2,
        moe_num_shared: int = 1,
        arch: str = "lgt",
        sliding_window: int = 512,
        local_global_ratio: int = 5,
        rope_theta_local: float = 10_000.0,
        rope_theta_global: float = 1_000_000.0,
        p_rope: float = 0.25,
        **_ignored,
    ):
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            max_length=max_length,
            attn_type=attn_type,
            block_size=block_size,
            use_time_cond=use_time_cond,
            learned_pos=False,
            tie_embeddings=tie_embeddings,
            arch=arch,
        )
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must divide n_heads={n_heads}")
        self.lgt = True
        self.sliding_window = sliding_window
        self.stack = LGTStack(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            n_kv_heads=n_kv_heads,
            ffn_type=ffn_type,
            moe_num_experts=moe_num_experts,
            moe_top_k=moe_top_k,
            moe_num_shared=moe_num_shared,
            use_time_cond=use_time_cond,
            sliding_window=sliding_window,
            local_global_ratio=local_global_ratio,
            rope_theta_local=rope_theta_local,
            rope_theta_global=rope_theta_global,
            p_rope=p_rope,
            causal_window=attn_type != "bidirectional",
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self._tie_head(self.lm_head)
        self.layers = self.stack.layers
        self.layer_is_global = self.stack.layer_is_global
        self.ln_out = self.stack.ln_out
        self.head = self.lm_head
