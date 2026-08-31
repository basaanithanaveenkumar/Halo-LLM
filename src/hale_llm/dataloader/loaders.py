from __future__ import annotations

import torch
from loguru import logger
from torch.utils.data import DataLoader

from hale_llm.dataloader.collate import get_collate
from hale_llm.dataloader.sources.huggingface import HuggingFaceTextDataset
from hale_llm.dataloader.tokenizer import setup_tokenizer


def make_dataloader(cfg, split: str = "train", **collate_kwargs) -> DataLoader:
    tok = setup_tokenizer(cfg.data.tokenizer_name)
    hf_split = cfg.data.train_split if split == "train" else cfg.data.val_split
    ds = HuggingFaceTextDataset(tok, cfg, split=hf_split)
    collate = get_collate(cfg.variant)

    def _collate(batch):
        tensors = torch.stack(list(batch), dim=0)
        return collate(tensors, tokenizer=tok, **collate_kwargs)

    logger.debug(
        "building dataloader split={} batch_size={} variant={}",
        split,
        cfg.train.batch_size,
        cfg.variant,
    )
    return DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=split == "train",
        drop_last=True,
        collate_fn=_collate,
    )
