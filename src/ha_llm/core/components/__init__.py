"""Reusable neural primitives: attention, FFN, norms, layers. No vocab, no training loop."""

from ha_llm.core.components.attention import (
    ATTENTION,
    build_attention,
    build_attn_mask,
    or_masks,
    register_attention,
    sliding_window_mask,
)
from ha_llm.core.components.dit import DiTBlock, DiTFinalLayer
from ha_llm.core.components.embeddings import SinusoidalTimeEmbedding
from ha_llm.core.components.ffn import FFN, build_ffn, register_ffn
from ha_llm.core.components.layer import TransformerLayer
from ha_llm.core.components.norm import RMSNorm

__all__ = [
    "ATTENTION",
    "FFN",
    "RMSNorm",
    "SinusoidalTimeEmbedding",
    "TransformerLayer",
    "DiTBlock",
    "DiTFinalLayer",
    "build_attention",
    "build_attn_mask",
    "build_ffn",
    "or_masks",
    "register_attention",
    "register_ffn",
    "sliding_window_mask",
]
