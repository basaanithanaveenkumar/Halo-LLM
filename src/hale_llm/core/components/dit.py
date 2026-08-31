"""DiT block (Peebles & Xie): pre-norm attention + FFN with AdaLN-Zero.

Zero-init modulation so the block starts as a residual transformer; gates, scales,
and shifts are learned from the timestep embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DiTBlock(nn.Module):
    def __init__(self, d_model: int, attention: nn.Module, ffn: nn.Module):
        super().__init__()
        self.attn = attention
        self.ff = ffn
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def _mod(self, t_emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        chunks = self.adaLN_modulation(t_emb).chunk(6, dim=-1)
        if chunks[0].dim() == 2:
            return tuple(c.unsqueeze(1) for c in chunks)
        return chunks

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if t_emb is None:
            t_emb = torch.zeros(x.size(0), x.size(-1), device=x.device, dtype=x.dtype)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._mod(t_emb)
        h = self.ln1(x) * (1 + scale_msa) + shift_msa
        x = x + gate_msa * self.attn(
            h, attn_mask=attn_mask, key_padding_mask=key_padding_mask, positions=positions
        )
        h = self.ln2(x) * (1 + scale_mlp) + shift_mlp
        return x + gate_mlp * self.ff(h)


class DiTFinalLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor | None) -> torch.Tensor:
        if t_emb is None:
            t_emb = torch.zeros(x.size(0), x.size(-1), device=x.device, dtype=x.dtype)
        shift, scale = self.adaLN_modulation(t_emb).chunk(2, dim=-1)
        if shift.dim() == 2:
            shift, scale = shift.unsqueeze(1), scale.unsqueeze(1)
        return self.head(self.ln(x) * (1 + scale) + shift)
