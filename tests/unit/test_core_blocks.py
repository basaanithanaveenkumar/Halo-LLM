import torch

from ha_llm.core.blocks.attention import build_attention, build_attn_mask
from ha_llm.core.blocks.ffn import build_ffn
from ha_llm.core.transformer import TransformerBackbone


def test_gqa_and_mqa_forward():
    x = torch.randn(2, 6, 16)
    gqa = build_attention("gqa", d_model=16, n_heads=4, n_kv_heads=2, dropout=0.0)
    mqa = build_attention("mqa", d_model=16, n_heads=4, dropout=0.0)
    mask = build_attn_mask("causal", 6, "cpu")
    assert gqa(x, attn_mask=mask).shape == x.shape
    assert mqa(x, attn_mask=mask).shape == x.shape


def test_moe_forward_shape():
    x = torch.randn(2, 5, 16)
    moe = build_ffn("moe", d_model=16, d_ff=32, moe_num_experts=4, moe_top_k=2, moe_num_shared=1)
    y = moe(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_backbone_gqa_moe():
    m = TransformerBackbone(
        vocab_size=32,
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        max_length=8,
        attn_impl="gqa",
        n_kv_heads=2,
        ffn_type="moe",
        moe_num_experts=2,
        moe_top_k=1,
        moe_num_shared=1,
    )
    x = torch.randint(0, 32, (2, 8))
    y = m(x)
    assert y.shape == (2, 8, 32)
