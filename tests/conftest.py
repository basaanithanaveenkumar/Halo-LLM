"""Tiny tokenizer so overfit tests do not hit the network."""

from __future__ import annotations

import torch
import pytest


class TinyTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"
    mask_token_id = 1
    vocab_size = 64

    def __len__(self):
        return self.vocab_size

    def __call__(
        self,
        text,
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt",
        add_special_tokens=False,
    ):
        ids = [2 + (ord(c) % (self.vocab_size - 2)) for c in text]
        if truncation:
            ids = ids[:max_length]
        if padding == "max_length":
            ids = ids + [self.pad_token_id] * (max_length - len(ids))
        t = torch.tensor(ids, dtype=torch.long)
        if return_tensors == "pt":
            return {"input_ids": t.unsqueeze(0)}
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(
            chr(32 + (i % 95)) for i in ids if not (skip_special_tokens and i in (0, 1))
        )


@pytest.fixture(autouse=True)
def _quiet_logging():
    from hale_llm.utils.logging import setup_logging

    setup_logging(level="WARNING", log_file=None, force=True)


@pytest.fixture
def tiny_tokenizer():
    return TinyTokenizer()
