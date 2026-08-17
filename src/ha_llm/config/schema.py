"""Compatibility barrel. Prefer `ha_llm.config` or `ha_llm.config.sections`."""

from ha_llm.config.load import load_config
from ha_llm.config.run import RunConfig
from ha_llm.config.sections import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    LoggingConfig,
    ModelConfig,
    SampleConfig,
    TrainConfig,
    VizConfig,
)

__all__ = [
    "RunConfig",
    "load_config",
    "ModelConfig",
    "TrainConfig",
    "DataConfig",
    "SampleConfig",
    "EvalConfig",
    "VizConfig",
    "LoggingConfig",
    "ExperimentConfig",
]
