"""RoPE and p-RoPE (lgt: full RoPE on local layers, p-RoPE on global)."""

from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    theta: float,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """q/k: (batch, heads, seq, head_dim). positions: (seq,)."""
    if rotary_dim <= 0:
        return q, k
    rotary_dim = rotary_dim - (rotary_dim % 2)
    device, dtype = q.device, q.dtype
    half = rotary_dim // 2
    idx = torch.arange(half, device=device, dtype=dtype)
    freqs = 1.0 / (theta ** (idx / half))
    pos = positions.to(device=device, dtype=dtype)
    angles = torch.outer(pos, freqs)
    cos = torch.cos(angles).repeat_interleave(2, dim=-1)[None, None, :, :]
    sin = torch.sin(angles).repeat_interleave(2, dim=-1)[None, None, :, :]

    def _rope(x: torch.Tensor) -> torch.Tensor:
        xr, rest = x[..., :rotary_dim], x[..., rotary_dim:]
        return torch.cat((xr * cos + rotate_half(xr) * sin, rest), dim=-1)

    return _rope(q), _rope(k)
