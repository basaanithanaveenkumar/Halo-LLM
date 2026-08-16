from pathlib import Path

import pytest
import yaml

from ha_llm.config.experiment import apply_experiment_layout
from ha_llm.config.schema import RunConfig


def _cfg(tmp_path: Path, **kwargs) -> RunConfig:
    return RunConfig(
        variant="autoregressive",
        experiment={"enabled": True, "root": str(tmp_path / "experiments"), "name": None},
        train={"resume": False, "checkpoint_path": "unused.pt"},
        logging={"log_file": "old.log", "tensorboard_dir": "old_tb"},
        viz={"output_dir": "old_viz"},
    )


def test_create_timestamped_run(tmp_path):
    cfg = _cfg(tmp_path)
    run_dir = apply_experiment_layout(cfg, create=True)
    assert run_dir is not None
    assert run_dir.parent.name == "autoregressive"
    assert run_dir.name.startswith("autoregressive_")
    assert (run_dir / "checkpoints").is_dir()
    assert (run_dir / "config.yaml").is_file()
    assert cfg.train.checkpoint_path == str(run_dir / "checkpoints" / "last.pt")
    assert cfg.logging.log_file == str(run_dir / "logs" / "train.log")
    assert cfg.logging.tensorboard_dir == str(run_dir / "tb")
    assert cfg.viz.output_dir == str(run_dir / "viz")
    latest = run_dir.parent / "latest"
    assert latest.is_symlink()
    assert latest.resolve() == run_dir.resolve()


def test_resume_uses_latest(tmp_path):
    first = apply_experiment_layout(_cfg(tmp_path), create=True)
    cfg = _cfg(tmp_path)
    cfg.train.resume = True
    again = apply_experiment_layout(cfg, create=True)
    assert again.resolve() == first.resolve()


def test_no_resume_creates_new_run(tmp_path):
    first = apply_experiment_layout(_cfg(tmp_path), create=True, name="run_1")
    second = apply_experiment_layout(_cfg(tmp_path), create=True, name="run_2")
    assert second.resolve() != first.resolve()
    assert (second.parent / "latest").resolve() == second.resolve()


def test_attach_named_run(tmp_path):
    created = apply_experiment_layout(_cfg(tmp_path), create=True, name="run_a")
    cfg = _cfg(tmp_path)
    attached = apply_experiment_layout(cfg, create=False, name="run_a")
    assert attached.resolve() == created.resolve()
    assert cfg.train.checkpoint_path.endswith("run_a/checkpoints/last.pt")


def test_attach_missing_name_raises(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(FileNotFoundError):
        apply_experiment_layout(cfg, create=False, name="does_not_exist")


def test_disabled_leaves_paths(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.experiment.enabled = False
    assert apply_experiment_layout(cfg, create=True) is None
    assert cfg.train.checkpoint_path == "unused.pt"


def test_copies_source_yaml_chain(tmp_path):
    base = tmp_path / "base.yaml"
    small = tmp_path / "small.yaml"
    base.write_text("variant: autoregressive\nmodel:\n  d_model: 32\n")
    small.write_text("inherits: base.yaml\ntrain:\n  steps: 3\n")
    cfg = _cfg(tmp_path)
    cfg.experiment.source_yaml = str(small)
    run_dir = apply_experiment_layout(cfg, create=True)
    copied = run_dir / "configs"
    assert (copied / "small.yaml").is_file()
    assert (copied / "base.yaml").is_file()
    dumped = (run_dir / "config.yaml").read_text()
    assert "autoregressive" in dumped
    parsed = yaml.safe_load(dumped)
    assert "source_yaml" not in parsed.get("experiment", {})
