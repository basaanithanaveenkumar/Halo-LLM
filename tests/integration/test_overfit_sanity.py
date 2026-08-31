from pathlib import Path

import pytest

import hale_llm.models  # noqa: F401
from hale_llm.config.schema import load_config
from hale_llm.training.trainer import train

REPO = Path(__file__).resolve().parents[2]

READY = ["autoregressive", "diffusion", "flow_matching", "block_diffusion"]
STUB = ["mtp"]


@pytest.mark.parametrize("variant", READY)
def test_overfit_loss_collapses(variant, tiny_tokenizer, tmp_path):
    cfg = load_config(REPO / "configs" / variant / "small.yaml")
    cfg.train.steps = 200
    cfg.train.epochs = None
    cfg.train.batch_size = 8
    cfg.train.log_every = 50
    cfg.train.checkpoint_path = str(tmp_path / "ckpt.pt")
    cfg.train.lr = 3e-3
    cfg.model.d_model = 64
    cfg.model.n_heads = 4
    cfg.model.n_layers = 2
    cfg.model.d_ff = 128
    cfg.model.max_length = 32 if variant != "block_diffusion" else 32
    if variant == "block_diffusion":
        cfg.model.block_size = 8
    cfg.data.overfit_text = "hello world hello world"
    cfg.data.n_overfit_copies = 32
    cfg.device = "cpu"
    cfg.logging.tensorboard = False
    cfg.logging.log_file = None
    cfg.logging.level = "WARNING"
    cfg.train.resume = False
    cfg.eval.every_n_epochs = None
    cfg.viz.enabled = False
    cfg.experiment.enabled = False

    _, _, losses = train(cfg, tokenizer=tiny_tokenizer)
    assert losses[-1] < losses[0] * 0.5, (losses[0], losses[-1])


@pytest.mark.parametrize("variant", STUB)
def test_mtp_stub_runs_a_few_steps(variant, tiny_tokenizer, tmp_path):
    cfg = load_config(REPO / "configs" / variant / "small.yaml")
    cfg.train.steps = 5
    cfg.train.epochs = None
    cfg.train.checkpoint_path = str(tmp_path / "ckpt.pt")
    cfg.model.d_model = 64
    cfg.model.n_heads = 4
    cfg.model.n_layers = 2
    cfg.model.d_ff = 128
    cfg.model.max_length = 32
    cfg.device = "cpu"
    cfg.logging.tensorboard = False
    cfg.logging.log_file = None
    cfg.logging.level = "WARNING"
    cfg.train.resume = False
    cfg.eval.every_n_epochs = None
    cfg.viz.enabled = False
    cfg.experiment.enabled = False
    _, _, losses = train(cfg, tokenizer=tiny_tokenizer)
    assert losses[-1] > 0
