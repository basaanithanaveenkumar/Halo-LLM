import torch

from hale_llm.core.components.attention.masks import build_attn_mask


def test_block_mask_shape():
    mask = build_attn_mask("block_causal", 16, "cpu", block_size=4)
    assert mask.shape == (32, 32)
    assert mask.dtype == torch.bool


def test_noised_cannot_attend_to_future_blocks():
    seq_len, block_size = 16, 4
    mask = build_attn_mask("block_causal", seq_len, "cpu", block_size=block_size)
    n = seq_len
    noised_b1 = n + block_size
    clean_b2 = 2 * block_size
    assert mask[noised_b1, clean_b2].item() is True
