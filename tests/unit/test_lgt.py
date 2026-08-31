import torch

from hale_llm.config.schema import load_config
from hale_llm.core.backbones.lgt import LGTBackbone
from hale_llm.core.components.attention import sliding_window_mask
from hale_llm.core.transformers.lgt import is_global_layer
from pathlib import Path


def test_global_layer_pattern_five_to_one():
    flags = [is_global_layer(i, 5) for i in range(16)]
    assert flags[5] and flags[11]
    assert flags.count(True) == 2
    assert not flags[0] and not flags[4]


def test_sliding_window_bidirectional():
    m = sliding_window_mask(8, window=2, device="cpu", causal=False)
    assert m[0, 3] and not m[0, 2]
    assert not m[4, 4]


def test_lgt_bidirectional_forward():
    m = LGTBackbone(
        vocab_size=32,
        d_model=16,
        n_heads=4,
        n_layers=6,
        d_ff=32,
        max_length=8,
        attn_type="bidirectional",
        use_time_cond=True,
        arch="lgt",
        n_kv_heads=2,
        ffn_type="geglu",
        sliding_window=4,
    )
    x = torch.randint(0, 32, (2, 8))
    t = torch.rand(2)
    y = m(x, t=t)
    assert y.shape == (2, 8, 32)
    assert torch.isfinite(y).all()
    assert m.layer_is_global[5]
    assert m.pos_emb is None
    assert m.layers[0].ln1_post is not None
    assert m.lgt


def test_lgt_block_causal_forward():
    m = LGTBackbone(
        vocab_size=32,
        d_model=16,
        n_heads=4,
        n_layers=2,
        d_ff=32,
        max_length=8,
        attn_type="block_causal",
        block_size=4,
        use_time_cond=True,
        arch="lgt",
        n_kv_heads=2,
        ffn_type="geglu",
    )
    tokens = torch.randint(0, 32, (2, 16))
    t = torch.rand(2, 16)
    y = m(tokens, t=t)
    assert y.shape == (2, 16, 32)


def test_diffusion_yaml_is_lgt():
    repo = Path(__file__).resolve().parents[2]
    for name in ("diffusion", "block_diffusion", "flow_matching"):
        cfg = load_config(repo / f"configs/{name}/small.yaml")
        assert cfg.model.arch == "lgt"
        assert cfg.model.attn_impl == "gqa"
        assert cfg.model.ffn_type == "geglu"
