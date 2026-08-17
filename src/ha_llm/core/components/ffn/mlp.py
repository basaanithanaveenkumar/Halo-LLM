from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class DenseMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class GeGLU(nn.Module):
    """Gated GELU FFN used by the LGT (local-global transformer) decoder."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w_in = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.w_out = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        gate, up = self.w_in(x).chunk(2, dim=-1)
        return self.w_out(self.drop(F.gelu(gate) * up))
