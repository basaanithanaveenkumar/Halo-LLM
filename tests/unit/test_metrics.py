import math

import torch

import ha_llm.models  # noqa: F401
from ha_llm.config.schema import RunConfig
from ha_llm.core.registry import instantiate_metrics
from ha_llm.evaluation.eval_loop import evaluate
from ha_llm.metrics import LossMetric, MetricContext, PerplexityMetric, TokenAccuracyMetric


def test_instantiate_skips_inapplicable_metric():
    metrics = instantiate_metrics(["loss", "token_accuracy", "masked_accuracy"], "autoregressive")
    names = [type(m).__name__ for m in metrics]
    assert "LossMetric" in names
    assert "TokenAccuracyMetric" in names
    assert not any("Masked" in n for n in names)


def test_instantiate_masked_for_diffusion():
    metrics = instantiate_metrics(["masked_accuracy"], "diffusion")
    assert len(metrics) == 1
    assert "Diffusion" in type(metrics[0]).__name__


def test_unknown_metric_raises():
    try:
        instantiate_metrics(["not_a_metric"], "autoregressive")
        assert False
    except KeyError:
        pass


def test_loss_and_perplexity_from_scalar():
    loss_m = LossMetric()
    ppl_m = PerplexityMetric()
    ctx = MetricContext(model=None, batch={}, loss=torch.tensor(2.0), variant="autoregressive")
    loss_m.update(ctx)
    ppl_m.update(ctx)
    assert abs(loss_m.compute()["loss"] - 2.0) < 1e-6
    assert abs(ppl_m.compute()["perplexity"] - math.exp(2.0)) < 1e-6


def test_token_accuracy_perfect():
    class _M(torch.nn.Module):
        def forward(self, batch):
            b, t = batch["labels"].shape
            v = 4
            logits = torch.zeros(b, t, v)
            logits.scatter_(-1, batch["labels"].clamp(min=0).unsqueeze(-1), 10.0)
            return logits

    labels = torch.tensor([[1, 2, -100]])
    ctx = MetricContext(
        model=_M(),
        batch={"labels": labels, "pad_token_id": 0},
        loss=torch.tensor(0.0),
        variant="autoregressive",
    )
    m = TokenAccuracyMetric()
    m.update(ctx)
    assert m.compute()["token_accuracy"] == 1.0


def test_evaluate_returns_dict(tiny_tokenizer):
    cfg = RunConfig(
        variant="autoregressive",
        train={"batch_size": 2, "steps": 1, "epochs": None},
        model={"d_model": 32, "n_heads": 4, "n_layers": 1, "d_ff": 64, "max_length": 8},
        eval={"metrics": ["loss", "perplexity", "token_accuracy"], "every_n_epochs": None},
        logging={"tensorboard": False, "log_file": None, "level": "WARNING"},
        device="cpu",
    )
    from ha_llm.core.registry import get_variant
    from ha_llm.dataloader.collate import make_overfit_loader

    model = get_variant("autoregressive")(vocab_size=len(tiny_tokenizer), cfg=cfg)
    loader = make_overfit_loader(
        "autoregressive", tiny_tokenizer, "hi", 8, 2, 4
    )
    out = evaluate(model, cfg, loader)
    assert "loss" in out and "perplexity" in out and "token_accuracy" in out
    assert out["perplexity"] > 1.0
    assert 0.0 <= out["token_accuracy"] <= 1.0
