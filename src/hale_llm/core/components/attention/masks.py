"""Sequence masks. True = blocked (matches nn.MultiheadAttention). Independent of QKV layout."""

from __future__ import annotations

from typing import Literal

import torch

AttnType = Literal["causal", "bidirectional", "block_causal"]


def build_attn_mask(
    attn_type: AttnType,
    seq_len: int,
    device: torch.device | str,
    block_size: int | None = None,
) -> torch.Tensor | None:
    if attn_type == "bidirectional":
        return None
    if attn_type == "causal":
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
        )
    if attn_type == "block_causal":
        if block_size is None:
            raise ValueError("block_causal attention requires block_size")
        if seq_len % block_size != 0:
            raise ValueError("seq_len must be divisible by block_size")
        n = seq_len
        blk = torch.arange(n, device=device) // block_size
        blk_combined = torch.cat([blk, blk])
        is_clean = torch.cat(
            [
                torch.ones(n, dtype=torch.bool, device=device),
                torch.zeros(n, dtype=torch.bool, device=device),
            ]
        )
        bi, bj = blk_combined.unsqueeze(1), blk_combined.unsqueeze(0)
        ci, cj = is_clean.unsqueeze(1), is_clean.unsqueeze(0)
        allowed = (
            (ci & cj & (bj <= bi))
            | ((~ci) & cj & (bj < bi))
            | ((~ci) & (~cj) & (bj == bi))
        )
        return ~allowed
    raise ValueError(f"unknown attn_type {attn_type!r}")
