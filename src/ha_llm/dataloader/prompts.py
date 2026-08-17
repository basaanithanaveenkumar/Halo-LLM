"""Validation-passage prompts for visualization."""

from __future__ import annotations

from loguru import logger

from ha_llm.dataloader.sources.huggingface import load_hf_split
from ha_llm.dataloader.sources.overfit import uses_overfit


def is_heading(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("=") and stripped.endswith("=")


def collect_dataset_passages(cfg, *, split: str | None = None, min_words: int = 8) -> list[str]:
    if uses_overfit(cfg):
        text = (cfg.data.overfit_text or "").strip()
        return [text] if text else []
    split = split or cfg.data.val_split
    raw = load_hf_split(cfg, split)
    field = cfg.data.text_field
    passages: list[str] = []
    for row in raw:
        text = (row.get(field) or "").strip()
        if not text or is_heading(text):
            continue
        if len(text.split()) < min_words:
            continue
        passages.append(text)
    logger.info("dataset viz passages split={} count={}", split, len(passages))
    return passages


def passage_prompt(text: str, n_words: int) -> str:
    words = text.split()
    if not words:
        return text
    return " ".join(words[: max(1, n_words)])


def dataset_prompt(cfg, *, seed: int = 0, passages: list[str] | None = None) -> str:
    n_words = int(cfg.viz.prompt_words)
    if getattr(cfg.viz, "prompt_source", "dataset") != "dataset":
        return cfg.sample.prompt
    pool = passages if passages is not None else collect_dataset_passages(cfg, min_words=n_words)
    if not pool:
        logger.warning("no dataset passages for viz; falling back to sample.prompt")
        return cfg.sample.prompt
    text = pool[int(seed) % len(pool)]
    return passage_prompt(text, n_words)
