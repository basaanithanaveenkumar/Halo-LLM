from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from loguru import logger
from transformers import AutoTokenizer

from ha_llm.core.registry import NamedRegistry

COLLATE_REGISTRY = NamedRegistry("collate")
COLLATES: dict[str, Callable[..., dict[str, Any]]] = COLLATE_REGISTRY.items


def register_collate(name: str):
    def deco(fn):
        COLLATE_REGISTRY.add(name, fn)
        return fn

    return deco


def get_collate(name: str):
    return COLLATE_REGISTRY.get(name)


def setup_tokenizer(name: str = "gpt2"):
    logger.debug("loading tokenizer {}", name)
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.eos_token
    if tok.mask_token is None:
        tok.add_special_tokens({"mask_token": "[MASK]"})
        logger.info("added [MASK] token to tokenizer {}", name)
    logger.debug("tokenizer {} vocab_size={}", name, len(tok))
    return tok


class TokenIdDataset(Dataset):
    def __init__(self, ids: torch.Tensor):
        self.ids = ids

    def __len__(self):
        return self.ids.size(0)

    def __getitem__(self, idx):
        return self.ids[idx]


def overfit_ids(tokenizer, text: str, max_length: int, n_copies: int) -> torch.Tensor:
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
        add_special_tokens=False,
    )
    row = enc["input_ids"].squeeze(0)
    return row.unsqueeze(0).repeat(n_copies, 1)


def make_overfit_loader(
    variant: str,
    tokenizer,
    text: str,
    max_length: int,
    batch_size: int,
    n_copies: int,
    **collate_kwargs,
) -> DataLoader:
    ids = overfit_ids(tokenizer, text, max_length, n_copies)
    ds = TensorDataset(ids)
    collate = get_collate(variant)

    def _collate(batch):
        tensors = torch.stack([b[0] for b in batch], dim=0)
        return collate(tensors, tokenizer=tokenizer, **collate_kwargs)

    logger.info(
        "overfit loader text={!r} copies={} seq_len={} batch_size={}",
        text,
        n_copies,
        max_length,
        min(batch_size, n_copies),
    )
    return DataLoader(ds, batch_size=min(batch_size, n_copies), shuffle=True, drop_last=True, collate_fn=_collate)
