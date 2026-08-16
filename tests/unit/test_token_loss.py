import torch

from ha_llm.config.schema import RunConfig
from ha_llm.losses.token import model_token_nll, token_nll


def test_focal_downweights_easy_examples():
    logits = torch.zeros(1, 2, 3)
    logits[0, 0, 1] = 8.0
    logits[0, 1, 2] = 0.1
    targets = torch.tensor([[1, 2]])
    ce = token_nll(logits, targets, reduction="none", loss_type="ce")
    fl = token_nll(logits, targets, reduction="none", loss_type="focal", focal_gamma=2.0, focal_alpha=1.0)
    assert fl[0, 0] < ce[0, 0]
    assert fl[0, 1] <= ce[0, 1] + 1e-6


def test_model_token_nll_reads_cfg():
    cfg = RunConfig(variant="autoregressive", train={"loss_type": "focal", "focal_gamma": 2.0})

    class M:
        def __init__(self):
            self.cfg = cfg

    logits = torch.randn(2, 4, 8)
    targets = torch.randint(0, 8, (2, 4))
    out = model_token_nll(M(), logits, targets, reduction="mean")
    assert out.ndim == 0
    assert torch.isfinite(out)
