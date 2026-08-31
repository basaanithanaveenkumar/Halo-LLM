from __future__ import annotations

import torch
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from hale_llm.dataloader.collate import get_collate


def uses_overfit(cfg) -> bool:
    return cfg.data.source == "overfit" or bool(cfg.data.overfit_text)


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
    return DataLoader(
        ds, batch_size=min(batch_size, n_copies), shuffle=True, drop_last=True, collate_fn=_collate
    )
