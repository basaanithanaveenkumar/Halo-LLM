import torch

from ha_llm.config.schema import ModelConfig
from ha_llm.core.backbones import BACKBONES, TransformerBackbone, build_backbone
from ha_llm.core.backbones.dit import DiTBackbone
from ha_llm.core.components import DiTBlock, build_attention, build_ffn


def test_backbone_registry():
    assert set(BACKBONES) >= {"default", "transformer", "lgt", "dit"}


def test_dit_block_zero_init_is_identity_residual():
    torch.manual_seed(0)
    d, n = 16, 4
    block = DiTBlock(
        d,
        attention=build_attention("mha", d_model=d, n_heads=n),
        ffn=build_ffn("mlp", d_model=d, d_ff=32),
    )
    x = torch.randn(2, 6, d)
    t = torch.randn(2, d)
    y = block(x, t_emb=t)
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-5)


def test_dit_backbone_forward():
    m = DiTBackbone(
        vocab_size=32,
        d_model=16,
        n_heads=4,
        n_layers=2,
        d_ff=32,
        max_length=8,
        attn_type="bidirectional",
        use_time_cond=True,
    )
    x = torch.randint(0, 32, (2, 8))
    t = torch.rand(2)
    y = m(x, t=t)
    assert y.shape == (2, 8, 32)
    assert torch.isfinite(y).all()


def test_build_backbone_swaps_dit():
    cfg = ModelConfig(arch="dit", attn_type="bidirectional", use_time_cond=True, d_model=16, n_heads=4, n_layers=1, d_ff=32, max_length=8)
    m = build_backbone(32, cfg)
    assert isinstance(m, DiTBackbone)
    y = m(torch.randint(0, 32, (1, 8)), t=torch.rand(1))
    assert y.shape == (1, 8, 32)


def test_from_config_respects_arch():
    cfg = ModelConfig(arch="dit", d_model=16, n_heads=4, n_layers=1, d_ff=32, max_length=8)
    m = TransformerBackbone.from_config(32, cfg)
    assert isinstance(m, DiTBackbone)
    cfg2 = ModelConfig(arch="default", d_model=16, n_heads=4, n_layers=1, d_ff=32, max_length=8)
    m2 = TransformerBackbone.from_config(32, cfg2)
    assert isinstance(m2, TransformerBackbone)
    assert not isinstance(m2, DiTBackbone)
