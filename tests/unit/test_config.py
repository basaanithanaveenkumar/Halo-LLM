from pathlib import Path

from hale_llm.config.schema import load_config


def test_bad_key_fails():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        from hale_llm.config.schema import RunConfig

        RunConfig.model_validate({"variant": "autoregressive", "model": {"not_a_field": 1}})


def test_resolve_schedule_epochs_wins():
    from hale_llm.config.schema import RunConfig
    from hale_llm.training.trainer import resolve_schedule

    cfg = RunConfig(variant="autoregressive", train={"epochs": 3, "steps": 999})
    n_epochs, total_steps = resolve_schedule(cfg, n_batches=10)
    assert n_epochs == 3
    assert total_steps == 30


def test_resolve_schedule_steps_only():
    from hale_llm.config.schema import RunConfig
    from hale_llm.training.trainer import resolve_schedule

    cfg = RunConfig(variant="autoregressive", train={"epochs": None, "steps": 25})
    n_epochs, total_steps = resolve_schedule(cfg, n_batches=10)
    assert total_steps == 25
    assert n_epochs == 3


def test_yaml_inherit():
    repo = Path(__file__).resolve().parents[2]
    cfg = load_config(repo / "configs/autoregressive/small.yaml")
    assert cfg.variant == "autoregressive"
    assert cfg.model.d_model == 384
    assert cfg.model.attn_type == "causal"
    assert cfg.model.attn_impl == "mha"
    assert cfg.model.ffn_type == "mlp"
    assert cfg.logging.tensorboard is True
    assert cfg.logging.level == "INFO"
    assert cfg.train.epochs == 2000
    assert cfg.train.steps is None
    assert cfg.train.batch_size == 32
    assert cfg.train.resume is False
    assert cfg.data.source == "huggingface"
    assert cfg.data.dataset == "Salesforce/wikitext"
    assert cfg.data.subset == "wikitext-2-raw-v1"
    assert cfg.data.train_split == "train"
    assert cfg.data.val_split == "validation"
    assert cfg.data.train_size is None
    assert cfg.data.overfit_text is None
    assert cfg.data.stride_words == 10
    assert cfg.device == "mps"
    assert "loss" in cfg.eval.metrics
    assert cfg.eval.every_n_epochs == 10
    assert cfg.viz.enabled is True
    assert cfg.viz.prompt_source == "dataset"
    assert cfg.viz.every_n_epochs is None
    assert cfg.experiment.enabled is True
    assert cfg.experiment.source_yaml.endswith("configs/autoregressive/small.yaml")
