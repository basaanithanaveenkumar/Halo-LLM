"""DiT token LM: learned positions + DiTStack + AdaLN final layer."""

from __future__ import annotations

from ha_llm.core.backbones.sequence import SequenceBackbone
from ha_llm.core.components import DiTFinalLayer
from ha_llm.core.components.attention import AttnType
from ha_llm.core.transformers.dit import DiTStack


class DiTBackbone(SequenceBackbone):
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
        use_time_cond: bool = True,
        block_size: int | None = None,
        tie_embeddings: bool = True,
        attn_impl: str = "mha",
        n_kv_heads: int | None = None,
        ffn_type: str = "mlp",
        moe_num_experts: int = 4,
        moe_top_k: int = 2,
        moe_num_shared: int = 1,
        arch: str = "dit",
        **_ignored,
    ):
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            max_length=max_length,
            attn_type=attn_type,
            block_size=block_size,
            use_time_cond=True,
            learned_pos=True,
            time_act="silu",
            tie_embeddings=tie_embeddings,
            arch=arch,
        )
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must divide n_heads={n_heads}")
        self.stack = DiTStack(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            attn_impl=attn_impl,
            n_kv_heads=n_kv_heads,
            ffn_type=ffn_type,
            moe_num_experts=moe_num_experts,
            moe_top_k=moe_top_k,
            moe_num_shared=moe_num_shared,
        )
        self.final = DiTFinalLayer(d_model, vocab_size)
        self._tie_head(self.final.head)
        self.lm_head = self.final
        self.blocks = self.stack.blocks

    def logits(self, hidden, t_emb):
        return self.final(hidden, t_emb)
