"""Hidden-state stacks. Operate on (batch, seq, d_model) — no tokenizer or LM head.

Use these from a VLM or world model: you supply embeddings, this module runs the decoder.
"""

from ha_llm.core.transformers.dit import DiTStack
from ha_llm.core.transformers.gpt import GPTStack
from ha_llm.core.transformers.lgt import LGTStack, is_global_layer

__all__ = ["GPTStack", "LGTStack", "DiTStack", "is_global_layer"]
