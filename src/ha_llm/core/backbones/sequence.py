"""Shared token embeddings, masks, and forward contract for sequence LMs."""

from __future__ import annotations

import torch
import torch.nn as nn

from ha_llm.core.components.attention import AttnType, build_attn_mask
from ha_llm.core.components.embeddings import SinusoidalTimeEmbedding


class SequenceBackbone(nn.Module):
    """Embed tokens → encode hidden states → logits.

    Subclasses set `self.stack` (hidden-state encoder) and `self.lm_head`.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_length: int,
        attn_type: AttnType = "causal",
        block_size: int | None = None,
        use_time_cond: bool = False,
        learned_pos: bool = True,
        time_act: str = "gelu",
        tie_embeddings: bool = True,
        arch: str = "transformer",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_length = max_length
        self.attn_type = attn_type
        self.block_size = block_size
        self.use_time_cond = use_time_cond
        self.arch = arch
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = None
        if learned_pos:
            self.pos_emb = nn.Parameter(torch.randn(1, max_length, d_model) * 0.02)
        self.time_embed = None
        if use_time_cond:
            act = nn.SiLU() if time_act == "silu" else nn.GELU()
            self.time_embed = nn.Sequential(
                SinusoidalTimeEmbedding(d_model),
                nn.Linear(d_model, d_model),
                act,
                nn.Linear(d_model, d_model),
            )
        self.stack: nn.Module | None = None
        self.lm_head: nn.Module | None = None
        self._tie = tie_embeddings

    def _tie_head(self, head: nn.Linear) -> None:
        if self._tie:
            head.weight = self.token_emb.weight

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = token_ids.shape
        hidden = self.token_emb(token_ids)
        if self.pos_emb is None:
            return hidden
        if seq_len <= self.max_length:
            return hidden + self.pos_emb[:, :seq_len, :]
        n = seq_len // 2
        pe = self.pos_emb[:, :n, :]
        return hidden + torch.cat([pe, pe], dim=1)

    def time_features(self, t: torch.Tensor | None) -> torch.Tensor | None:
        if not self.use_time_cond or t is None or self.time_embed is None:
            return None
        return self.time_embed(t)

    def positions(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self.attn_type == "block_causal" and seq_len > self.max_length:
            n = seq_len // 2
            pos = torch.arange(n, device=device)
            return torch.cat([pos, pos], dim=0)
        return torch.arange(min(seq_len, self.max_length), device=device)

    def attention_inputs(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
        pos_ids_len: int | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        _, seq_len = token_ids.shape
        pos_len = pos_ids_len or min(seq_len, self.max_length)
        if pos_len > self.max_length and self.attn_type != "block_causal":
            raise ValueError(f"seq {pos_len} > max_length {self.max_length}")
        key_padding = (attention_mask == 0) if attention_mask is not None else None
        if attn_mask is None:
            stream_len = seq_len // 2 if self.attn_type == "block_causal" else seq_len
            attn_mask = build_attn_mask(
                self.attn_type,
                stream_len,
                token_ids.device,
                block_size=self.block_size,
            )
        return attn_mask, key_padding

    def encode(
        self,
        hidden: torch.Tensor,
        t_emb: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if self.stack is None:
            raise RuntimeError("SequenceBackbone.stack was not set")
        return self.stack(
            hidden,
            t_emb=t_emb,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            positions=positions,
        )

    def logits(self, hidden: torch.Tensor, t_emb: torch.Tensor | None) -> torch.Tensor:
        if self.lm_head is None:
            raise RuntimeError("SequenceBackbone.lm_head was not set")
        return self.lm_head(hidden)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        pos_ids_len: int | None = None,
    ) -> torch.Tensor:
        hidden = self.embed(token_ids)
        t_emb = self.time_features(t)
        attn_mask, key_padding = self.attention_inputs(
            token_ids, attention_mask, attn_mask, pos_ids_len
        )
        hidden = self.encode(
            hidden,
            t_emb,
            attn_mask,
            key_padding,
            self.positions(token_ids.size(1), token_ids.device),
        )
        return self.logits(hidden, t_emb)

    @classmethod
    def from_config(cls, vocab_size: int, model_cfg, **overrides):
        from ha_llm.core.backbones.factory import build_backbone

        return build_backbone(vocab_size, model_cfg, **overrides)
