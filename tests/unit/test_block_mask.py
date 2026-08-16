import torch

from halodiffusionllm.models.block_diffusion import build_block_diffusion_mask


def test_block_mask_shape():
    mask = build_block_diffusion_mask(16, 4, "cpu")
    assert mask.shape == (32, 32)
    assert mask.dtype == torch.bool


def test_noised_cannot_attend_to_future_blocks():
    seq_len, block_size = 16, 4
    mask = build_block_diffusion_mask(seq_len, block_size, "cpu")
    N = seq_len
    # noised stream index for block 1, position 0
    noised_b1 = N + block_size
    # clean token in block 2 (future)
    clean_b2 = 2 * block_size
    assert mask[noised_b1, clean_b2].item() is True  # blocked
