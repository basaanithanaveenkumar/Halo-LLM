"""Convenience entry: config → checkpoint → text."""

from __future__ import annotations

from hale_llm.config.schema import RunConfig
from hale_llm.inference.session import GenerationResult, load_session


def generate(
    cfg: RunConfig,
    prompt: str | None = None,
    *,
    checkpoint: str | None = None,
    return_history: bool = False,
    **kwargs,
) -> GenerationResult:
    return load_session(cfg, checkpoint).generate(prompt, return_history=return_history, **kwargs)
