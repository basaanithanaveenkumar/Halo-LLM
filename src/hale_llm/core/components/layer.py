"""Pre-norm layer: injected attention + FFN (open for new impls, closed for callers)."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_llm.core.components.norm import RMSNorm


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        attention: nn.Module,
        ffn: nn.Module,
        use_time_cond: bool,
        norm: str = "layer",
        post_norm: bool = False,
    ):
        super().__init__()
        self.attn = attention
        self.ff = ffn
        self.use_time_cond = use_time_cond
        self.post_norm = post_norm
        self.ln1 = RMSNorm(d_model) if norm == "rms" else nn.LayerNorm(d_model)
        self.ln2 = RMSNorm(d_model) if norm == "rms" else nn.LayerNorm(d_model)
        self.ln1_post = RMSNorm(d_model) if post_norm else None
        self.ln2_post = RMSNorm(d_model) if post_norm else None
        if use_time_cond:
            self.time_proj = nn.Linear(d_model, 4 * d_model)
            nn.init.zeros_(self.time_proj.weight)
            nn.init.zeros_(self.time_proj.bias)

    def _modulate(self, h: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        if scale.dim() == 2:
            scale, shift = scale.unsqueeze(1), shift.unsqueeze(1)
        return h * (1 + scale) + shift

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        s1 = sh1 = s2 = sh2 = None
        if self.use_time_cond and t_emb is not None:
            s1, sh1, s2, sh2 = self.time_proj(t_emb).chunk(4, dim=-1)
        h = self.ln1(x)
        if s1 is not None:
            h = self._modulate(h, s1, sh1)
        x = x + self.attn(
            h, attn_mask=attn_mask, key_padding_mask=key_padding_mask, positions=positions
        )
        if self.ln1_post is not None:
            x = self.ln1_post(x)
        h = self.ln2(x)
        if s2 is not None:
            h = self._modulate(h, s2, sh2)
        x = x + self.ff(h)
        if self.ln2_post is not None:
            x = self.ln2_post(x)
        return x
