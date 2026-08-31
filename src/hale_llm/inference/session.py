"""Load a checkpointed variant and run the registered sampler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger
from torch import nn

from hale_llm.config.schema import RunConfig
from hale_llm.core.checkpoint import CheckpointStore
from hale_llm.core.registry import get_sampler, get_variant
from hale_llm.core.tensors import log_model_summary
from hale_llm.dataloader.tokenizer import setup_tokenizer
from hale_llm.inference.encode import encode_prompt
from hale_llm.utils.device import get_device


@dataclass
class GenerationResult:
    token_ids: torch.Tensor
    text: str
    prompt: str
    history: list | None = None


@dataclass
class InferenceSession:
    model: nn.Module
    tokenizer: Any
    cfg: RunConfig
    device: str

    def generate(
        self,
        prompt: str | None = None,
        *,
        return_history: bool = False,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        sampling_steps: int | None = None,
    ) -> GenerationResult:
        prompt = prompt if prompt is not None else self.cfg.sample.prompt
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.cfg.sample.max_new_tokens
        temperature = temperature if temperature is not None else self.cfg.sample.temperature
        sampling_steps = sampling_steps if sampling_steps is not None else self.cfg.sample.sampling_steps
        sampler = get_sampler(self.cfg.variant)
        self.model.eval()
        ids = encode_prompt(self.tokenizer, prompt, self.model, self.device)
        logger.info("inference variant={} prompt={!r} return_history={}", self.cfg.variant, prompt, return_history)
        out = sampler(
            self.model,
            self.tokenizer,
            ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            sampling_steps=sampling_steps,
            return_history=return_history,
        )
        history = None
        if return_history:
            token_ids, history = out
        else:
            token_ids = out
        text = self.tokenizer.decode(token_ids[0].tolist(), skip_special_tokens=True)
        logger.info("generated: {}", text)
        return GenerationResult(token_ids=token_ids, text=text, prompt=prompt, history=history)


def session_from_model(model: nn.Module, tokenizer, cfg: RunConfig, device: str | None = None) -> InferenceSession:
    device = device or get_device(cfg.device)
    return InferenceSession(model=model.to(device), tokenizer=tokenizer, cfg=cfg, device=device)


def load_session(
    cfg: RunConfig,
    checkpoint_path: str | None = None,
    *,
    require_checkpoint: bool = False,
    tokenizer=None,
    checkpoints: CheckpointStore | None = None,
) -> InferenceSession:
    """Load weights if the checkpoint exists.

    Sample/viz default to an untrained model when the configured path is missing.
    Pass ``require_checkpoint=True`` (eval) or an explicit ``checkpoint_path`` that
    does not exist to keep the old FileNotFoundError.
    """
    import hale_llm.models  # noqa: F401

    device = get_device(cfg.device)
    tokenizer = tokenizer or setup_tokenizer(cfg.data.tokenizer_name)
    store = checkpoints or CheckpointStore()
    path = checkpoint_path or cfg.train.checkpoint_path
    model_cls = get_variant(cfg.variant)

    if store.exists(path):
        ckpt = store.load(path, device)
        model = model_cls(vocab_size=ckpt["vocab_size"], cfg=cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("inference session variant={} device={} ckpt={}", cfg.variant, device, path)
        log_model_summary(model, title=f"model summary  variant={cfg.variant}")
        return InferenceSession(model=model, tokenizer=tokenizer, cfg=cfg, device=device)

    explicit_missing = checkpoint_path is not None
    if require_checkpoint or explicit_missing:
        store.load(path, device)

    logger.warning(
        "no checkpoint at {}; using untrained {}. "
        "Train first, or pass --checkpoint PATH",
        path,
        cfg.variant,
    )
    model = model_cls(vocab_size=len(tokenizer), cfg=cfg).to(device)
    log_model_summary(model, title=f"model summary  variant={cfg.variant}")
    return InferenceSession(model=model, tokenizer=tokenizer, cfg=cfg, device=device)
