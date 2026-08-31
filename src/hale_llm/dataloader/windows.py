"""Word-stride packing of raw text into (N, max_length) token windows."""

from __future__ import annotations

import torch


def encode_words(tokenizer, words: list[str]) -> tuple[list[int], list[int]]:
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
    if stride_words < 1:
        raise ValueError(f"stride_words must be >= 1, got {stride_words}")
    words = text.split()
    if not words:
        return torch.zeros(0, max_length, dtype=torch.long)
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    ids, starts = encode_words(tokenizer, words)
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
