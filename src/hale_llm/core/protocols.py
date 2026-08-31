"""Structural types so training/eval depend on interfaces, not concrete variants."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class LossFn(Protocol):
    def __call__(self, model: nn.Module, batch: dict[str, Any]) -> torch.Tensor: ...


@runtime_checkable
class SamplerFn(Protocol):
    def __call__(self, model: nn.Module, tokenizer: Any, prompt_ids: torch.Tensor, **kwargs: Any) -> Any: ...


@runtime_checkable
class AttentionBlock(Protocol):
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


@runtime_checkable
class FeedForwardBlock(Protocol):
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
