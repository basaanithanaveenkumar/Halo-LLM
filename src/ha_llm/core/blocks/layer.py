"""Pre-norm layer: injected attention + FFN (open for new impls, closed for callers)."""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        attention: nn.Module,
        ffn: nn.Module,
        use_time_cond: bool,
    ):
        super().__init__()
        self.attn = attention
        self.ff = ffn
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.use_time_cond = use_time_cond
        if use_time_cond:
            self.time_proj = nn.Linear(d_model, 4 * d_model)
            nn.init.zeros_(self.time_proj.weight)
            nn.init.zeros_(self.time_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_time_cond and t_emb is not None:
            s1, sh1, s2, sh2 = self.time_proj(t_emb).chunk(4, dim=-1)
            if s1.dim() == 2:
                s1, sh1, s2, sh2 = (t.unsqueeze(1) for t in (s1, sh1, s2, sh2))
            h = self.ln1(x) * (1 + s1) + sh1
        else:
            h = self.ln1(x)
        x = x + self.attn(h, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        if self.use_time_cond and t_emb is not None:
            h = self.ln2(x) * (1 + s2) + sh2
        else:
            h = self.ln2(x)
        return x + self.ff(h)
