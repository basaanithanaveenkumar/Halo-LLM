"""Variant-agnostic viz entry used by CLI and Trainer. No models_* imports."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from ha_llm.core.registry import get_sampler
from ha_llm.visualization.unmasking import visualize_generation


def run_visualization(
    model,
    tokenizer,
    cfg,
    device: str,
    output_gif: str,
    *,
    train: bool = False,
    prompt: str | None = None,
) -> str:
    sampler = get_sampler(cfg.variant)
    if prompt is None:
        from ha_llm.dataloader.dataset import dataset_prompt

        prompt = dataset_prompt(cfg) if cfg.viz.prompt_source == "dataset" else cfg.sample.prompt
    Path(output_gif).parent.mkdir(parents=True, exist_ok=True)
    if train:
        sampling_steps = cfg.viz.sampling_steps or cfg.sample.sampling_steps
        max_new_tokens = cfg.viz.max_new_tokens or cfg.sample.max_new_tokens
    else:
        sampling_steps = cfg.sample.sampling_steps
        max_new_tokens = cfg.sample.max_new_tokens
    logger.info(
        "writing viz gif={} variant={} prompt={!r} sampling_steps={} max_new_tokens={}",
        output_gif,
        cfg.variant,
        prompt,
        sampling_steps,
        max_new_tokens,
    )
    visualize_generation(
        model,
        tokenizer,
        sampler,
        prompt=prompt,
        device=device,
        max_new_tokens=max_new_tokens,
        sampling_steps=sampling_steps,
        temperature=cfg.sample.temperature,
        output_gif=output_gif,
        variant=cfg.variant,
    )
    return output_gif
