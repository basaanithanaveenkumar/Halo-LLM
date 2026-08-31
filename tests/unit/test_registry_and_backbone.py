import torch

import hale_llm.core.optimizers  # noqa: F401
import hale_llm.models  # noqa: F401
from hale_llm.core.registry import MODELS, get_model, get_optimizer
from hale_llm.core.backbones import TransformerBackbone
from hale_llm.core.components import build_attn_mask


def test_registry_has_all_models():
    assert set(MODELS) >= {
        "autoregressive",
        "diffusion",
        "block_diffusion",
        "flow_matching",
        "mtp",
    }


def test_causal_mask_upper_triangle():
    m = build_attn_mask("causal", 4, "cpu")
    assert m[0, 1]
    assert not m[1, 0]


def test_backbone_causal_forward():
    m = TransformerBackbone(vocab_size=32, d_model=16, n_heads=4, n_layers=1, d_ff=32, max_length=8)
    x = torch.randint(0, 32, (2, 8))
    y = m(x)
    assert y.shape == (2, 8, 32)


def test_format_model_summary_lists_modules():
    import torch.nn as nn

    from hale_llm.core.tensors import format_model_summary

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))

    text = format_model_summary(Wrapper(), title="model summary")
    assert "TOTAL" in text
    assert "Trainable" in text
    assert "backbone" in text


def test_get_model_unknown():
    try:
        get_model("nope")
        assert False
    except KeyError:
        pass


def test_optimizer_registry_has_builtins():
    from hale_llm.core.registry import OPTIMIZERS

    assert {"adam", "adamw", "sgd", "muon"} <= set(OPTIMIZERS)
    assert get_optimizer("adamw") is get_optimizer("AdamW")


def test_build_adamw_has_torch_api():
    import torch.nn as nn

    from hale_llm.core.optimizers import build_optimizer

    model = nn.Linear(4, 2)
    opt = build_optimizer("adamw", model.parameters(), {"lr": 1e-3, "weight_decay": 0.0})
    loss = model(torch.ones(3, 4)).sum()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    assert "lr" in opt.param_groups[0]
    state = opt.state_dict()
    opt.load_state_dict(state)
