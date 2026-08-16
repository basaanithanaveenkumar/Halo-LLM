import torch

from halodiffusionllm.models.embeddings import SinusoidalTimeEmbedding


def test_sinusoidal_time_embedding_batch():
    emb = SinusoidalTimeEmbedding(64)
    out = emb(torch.zeros(4))
    assert out.shape == (4, 64)


def test_sinusoidal_time_embedding_per_token():
    emb = SinusoidalTimeEmbedding(64)
    out = emb(torch.zeros(2, 16))
    assert out.shape == (2, 16, 64)
