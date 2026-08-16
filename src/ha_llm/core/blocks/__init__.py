"""Composable transformer sub-blocks. Core never imports models_*."""

from ha_llm.core.blocks.attention import build_attention, build_attn_mask
from ha_llm.core.blocks.ffn import build_ffn
from ha_llm.core.blocks.layer import TransformerLayer

__all__ = ["build_attention", "build_attn_mask", "build_ffn", "TransformerLayer"]
