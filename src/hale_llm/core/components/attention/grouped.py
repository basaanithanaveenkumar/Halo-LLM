"""Grouped-query family (HaloBlocks GQA/MQA). n_kv_heads=n_heads is MHA; n_kv_heads=1 is MQA."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_llm.core.components.attention.rope import apply_rope
from hale_llm.core.components.attention.scoring import apply_attn_masks
from hale_llm.core.components.norm import RMSNorm


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float = 0.0,
        rope_theta: float | None = None,
        rotary_dim: int | None = None,
        qk_norm: bool = False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must divide n_heads={n_heads}")
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads={n_heads} must divide n_kv_heads={n_kv_heads}")
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.group_size = n_heads // n_kv_heads
        self.rope_theta = rope_theta
        self.rotary_dim = rotary_dim if rotary_dim is not None else self.head_dim
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else None

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.wq(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.rope_theta is not None:
            if positions is None:
                positions = torch.arange(s, device=x.device)
            q, k = apply_rope(q, k, positions, self.rope_theta, self.rotary_dim)
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = apply_attn_masks(scores, attn_mask, key_padding_mask)
        w = self.drop(F.softmax(scores, dim=-1))
        out = torch.matmul(w, v).transpose(1, 2).contiguous().view(b, s, -1)
        return self.wo(out)
