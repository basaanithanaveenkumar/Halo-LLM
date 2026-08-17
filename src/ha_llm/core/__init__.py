"""Library core: components, transformer stacks, and token backbones.

Intended reuse (VLM, world models):
    from ha_llm.core.transformers import LGTStack, GPTStack, DiTStack
    hidden = LGTStack(...)(vision_and_text_tokens, attn_mask=mask)

    from ha_llm.core.backbones import build_backbone
    logits = build_backbone(vocab_size, cfg.model)(token_ids, t=t)

Training plugins (models/losses/samplers) stay in `ha_llm.core.registry`.
"""

from ha_llm.core.backbones import (
    BACKBONES,
    DiTBackbone,
    GPTBackbone,
    LGTBackbone,
    TransformerBackbone,
    build_backbone,
)
from ha_llm.core.checkpoint import CheckpointStore, load_checkpoint, save_checkpoint
from ha_llm.core.components import build_attn_mask
from ha_llm.core.registry import (
    NamedRegistry,
    get_loss,
    get_model,
    get_sampler,
    get_variant,
)
from ha_llm.core.transformers import DiTStack, GPTStack, LGTStack

__all__ = [
    "BACKBONES",
    "build_backbone",
    "GPTBackbone",
    "LGTBackbone",
    "TransformerBackbone",
    "DiTBackbone",
    "GPTStack",
    "LGTStack",
    "DiTStack",
    "build_attn_mask",
    "NamedRegistry",
    "CheckpointStore",
    "get_model",
    "get_variant",
    "get_loss",
    "get_sampler",
    "save_checkpoint",
    "load_checkpoint",
]
