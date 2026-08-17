from __future__ import annotations

import torch
from datasets import load_dataset
from loguru import logger
from torch.utils.data import Dataset

from ha_llm.dataloader.windows import sliding_windows


def load_hf_split(cfg, split: str):
    data = cfg.data
    name = data.subset if data.subset else None
    kwargs: dict = {"split": split}
    if data.cache_dir:
        kwargs["cache_dir"] = data.cache_dir
    if name:
        raw = load_dataset(data.dataset, name, **kwargs)
    else:
        raw = load_dataset(data.dataset, **kwargs)
    size = data.train_size if split == data.train_split else data.val_size
    if size is not None:
        raw = raw.select(range(min(size, len(raw))))
    logger.info(
        "hf dataset={} subset={} split={} rows={} text_field={}",
        data.dataset,
        data.subset,
        split,
        len(raw),
        data.text_field,
    )
    return raw


def corpus_from_split(raw, text_field: str = "text") -> str:
    parts = []
    for row in raw:
        text = (row.get(text_field) or "").strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


class HuggingFaceTextDataset(Dataset):
    def __init__(self, tokenizer, cfg, split: str):
        self.tokenizer = tokenizer
        self.max_length = cfg.model.max_length
        self.stride_words = cfg.data.stride_words
        raw = load_hf_split(cfg, split)
        corpus = corpus_from_split(raw, cfg.data.text_field)
        self.ids = sliding_windows(tokenizer, corpus, self.max_length, self.stride_words)
        logger.info(
            "windows={} max_length={} stride_words={}",
            len(self.ids),
            self.max_length,
            self.stride_words,
        )

    def __len__(self):
        return int(self.ids.size(0))

    def __getitem__(self, idx):
        return self.ids[idx]
