"""Autoregressive LM — reference variant.

Pattern every other variant follows:
  model.py   @register_variant  wraps TransformerBackbone, defines forward(batch)
  loss.py    @register_loss     (model, batch) -> scalar
  sample.py  @register_sampler  generation
  collate registered here so data/ never imports this package by name
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ha_llm.core.registry import register_variant
from ha_llm.core.transformer import TransformerBackbone
from ha_llm.dataloader.collate import register_collate


@register_collate("autoregressive")
def collate_autoregressive(input_ids: torch.Tensor, tokenizer, **_):
    """Shifted next-token targets. Pad positions are ignored in the loss via labels=-100."""
    pad_id = tokenizer.pad_token_id
    attn = (input_ids != pad_id).long()
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100
    labels[input_ids == pad_id] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
        "pad_token_id": pad_id,
    }


@register_variant("autoregressive")
class AutoregressiveLM(nn.Module):
    def __init__(self, vocab_size: int, cfg):
        super().__init__()
        self.backbone = TransformerBackbone.from_config(vocab_size, cfg.model)
        self.vocab_size = vocab_size
        self.cfg = cfg

    def forward(self, batch: dict) -> torch.Tensor:
        return self.backbone(batch["input_ids"], attention_mask=batch.get("attention_mask"))
