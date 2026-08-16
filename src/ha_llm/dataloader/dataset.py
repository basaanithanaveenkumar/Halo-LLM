"""Shared dataset reading. Variant-specific batch shaping lives in collate.py."""

from __future__ import annotations

import torch
from datasets import load_dataset
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from ha_llm.dataloader.collate import get_collate, setup_tokenizer


def _encode_words(tokenizer, words: list[str]) -> tuple[list[int], list[int]]:
    """Tokenize word-by-word so windows can stride on word boundaries.

    Continuation words are prefixed with a space (GPT-2 BPE).
    """
    ids: list[int] = []
    starts: list[int] = []
    unk = getattr(tokenizer, "unk_token_id", None) or getattr(tokenizer, "eos_token_id", 0) or 0
    for i, word in enumerate(words):
        starts.append(len(ids))
        piece = word if i == 0 else f" {word}"
        toks = tokenizer.encode(piece, add_special_tokens=False)
        if not toks:
            toks = [unk]
        ids.extend(toks)
    return ids, starts


def sliding_windows(
    tokenizer,
    text: str,
    max_length: int,
    stride_words: int = 10,
) -> torch.Tensor:
    """Pack ``text`` into ``(N, max_length)`` windows, advancing ``stride_words`` each time."""
    if stride_words < 1:
        raise ValueError(f"stride_words must be >= 1, got {stride_words}")
    words = text.split()
    if not words:
        return torch.zeros(0, max_length, dtype=torch.long)
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    ids, starts = _encode_words(tokenizer, words)
    n_words = len(words)
    rows: list[list[int]] = []
    i = 0
    while i < n_words:
        start = starts[i]
        if start >= len(ids):
            break
        chunk = ids[start : start + max_length]
        if not chunk:
            break
        if len(chunk) < max_length:
            chunk = chunk + [pad_id] * (max_length - len(chunk))
        rows.append(chunk)
        if start + max_length >= len(ids):
            break
        i += stride_words
    if not rows:
        return torch.zeros(0, max_length, dtype=torch.long)
    return torch.tensor(rows, dtype=torch.long)


def _corpus_from_split(raw) -> str:
    parts = []
    for row in raw:
        text = (row.get("text") or "").strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


class WikiTextDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        split: str = "train",
        size: int | None = 2048,
        max_length: int = 64,
        stride_words: int = 10,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride_words = stride_words
        raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
        if size is not None:
            raw = raw.select(range(min(size, len(raw))))
        corpus = _corpus_from_split(raw)
        self.ids = sliding_windows(tokenizer, corpus, max_length, stride_words)
        logger.info(
            "WikiText-2 split={} rows={} windows={} max_length={} stride_words={}",
            split,
            len(raw),
            len(self.ids),
            max_length,
            stride_words,
        )

    def __len__(self):
        return int(self.ids.size(0))

    def __getitem__(self, idx):
        return self.ids[idx]


def _is_heading(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("=") and stripped.endswith("=")


def collect_dataset_passages(cfg, *, split: str = "validation", min_words: int = 8) -> list[str]:
    """Non-empty WikiText paragraphs long enough to use as viz prompts."""
    if cfg.data.overfit_text:
        return [cfg.data.overfit_text.strip()]
    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    size = cfg.data.val_size if split == "validation" else cfg.data.train_size
    if size is not None:
        raw = raw.select(range(min(size, len(raw))))
    passages: list[str] = []
    for row in raw:
        text = (row.get("text") or "").strip()
        if not text or _is_heading(text):
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
    """Prefix from a WikiText paragraph, or ``sample.prompt`` if none are available."""
    n_words = int(cfg.viz.prompt_words)
    if getattr(cfg.viz, "prompt_source", "dataset") != "dataset":
        return cfg.sample.prompt
    pool = passages if passages is not None else collect_dataset_passages(cfg, min_words=n_words)
    if not pool:
        logger.warning("no dataset passages for viz; falling back to sample.prompt")
        return cfg.sample.prompt
    text = pool[int(seed) % len(pool)]
    return passage_prompt(text, n_words)


def make_dataloader(cfg, split: str = "train", **collate_kwargs) -> DataLoader:
    tok = setup_tokenizer(cfg.data.tokenizer_name)
    size = cfg.data.train_size if split == "train" else cfg.data.val_size
    ds = WikiTextDataset(
        tok,
        split=split,
        size=size,
        max_length=cfg.model.max_length,
        stride_words=cfg.data.stride_words,
    )
    collate = get_collate(cfg.variant)

    def _collate(batch):
        tensors = torch.stack(list(batch), dim=0)
        return collate(tensors, tokenizer=tok, **collate_kwargs)

    logger.debug("building dataloader split={} batch_size={} variant={}", split, cfg.train.batch_size, cfg.variant)
    return DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=split == "train",
        drop_last=True,
        collate_fn=_collate,
    )
