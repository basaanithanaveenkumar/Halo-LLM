from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.0
    max_length: int = 64
    attn_type: Literal["causal", "bidirectional", "block_causal"] = "causal"
    attn_impl: Literal["mha", "gqa", "mqa"] = "mha"
    n_kv_heads: int | None = None
    ffn_type: Literal["mlp", "moe"] = "mlp"
    moe_num_experts: int = 4
    moe_top_k: int = 2
    moe_num_shared: int = 1
    use_time_cond: bool = False
    block_size: int | None = None
    n_mtp_heads: int = 2


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: int | None = 300
    epochs: int | None = None
    lr: float = 1e-3
    batch_size: int = 8
    weight_decay: float = 0.0
    log_every: int = 50
    grad_clip: float = 1.0
    seed: int = 0
    checkpoint_path: str = "checkpoints/last.pt"
    checkpoint_every_epoch: bool = True
    resume: bool = True
    loss_type: Literal["ce", "focal"] = "ce"
    focal_gamma: float = 2.0
    focal_alpha: float = 1.0


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tokenizer_name: str = "gpt2"
    train_size: int | None = 2048
    val_size: int | None = 256
    overfit_text: str | None = "hello world this is a tiny overfit sentence"
    n_overfit_copies: int = 64
    add_special_tokens: bool = False
    stride_words: int = 10


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_file: str | None = "logs/ha_llm.log"
    tensorboard: bool = True
    tensorboard_dir: str = "runs"


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metrics: list[str] = Field(
        default_factory=lambda: [
            "loss",
            "perplexity",
            "bits_per_token",
            "token_accuracy",
            "masked_accuracy",
        ]
    )
    every_n_epochs: int | None = 1
    max_batches: int | None = None


class SampleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_new_tokens: int = 32
    temperature: float = 1.0
    sampling_steps: int = 32
    eps: float = 1e-3
    prompt: str = "hello"


class VizConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    every_n_steps: int | None = 500
    every_n_epochs: int | None = None
    sampling_steps: int | None = 8
    max_new_tokens: int | None = None
    prompt_source: Literal["dataset", "config"] = "dataset"
    prompt_words: int = 8
    output_dir: str = "data/viz"


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    root: str = "data/experiments"
    name: str | None = None
    source_yaml: str | None = None


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variant: str
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    sample: SampleConfig = Field(default_factory=SampleConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    viz: VizConfig = Field(default_factory=VizConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    device: str | None = None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k == "inherits":
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> RunConfig:
    from loguru import logger

    path = Path(path).resolve()
    logger.debug("loading config from {}", path)
    if not path.exists():
        logger.error("config file not found: {}", path)
        raise FileNotFoundError(path)
    raw = yaml.safe_load(path.read_text()) or {}
    if "inherits" in raw:
        parent = (path.parent / raw["inherits"]).resolve()
        logger.debug("inheriting from {}", parent)
        parent_raw = yaml.safe_load(parent.read_text()) or {}
        if "inherits" in parent_raw:
            grand = (parent.parent / parent_raw["inherits"]).resolve()
            grand_raw = yaml.safe_load(grand.read_text()) or {}
            parent_raw = _deep_merge(grand_raw, parent_raw)
        raw = _deep_merge(parent_raw, raw)
    try:
        cfg = RunConfig.model_validate(raw)
    except Exception:
        logger.exception("invalid config {}", path)
        raise
    cfg.experiment.source_yaml = str(path)
    logger.info("loaded config variant={} from {}", cfg.variant, path)
    return cfg
