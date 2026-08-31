"""TensorBoard SummaryWriter helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from torch.utils.tensorboard import SummaryWriter


def make_writer(log_dir: str | None, enabled: bool = True, run_name: str | None = None) -> SummaryWriter | None:
    if not enabled:
        logger.debug("tensorboard disabled")
        return None
    if not log_dir:
        logger.warning("tensorboard enabled but no log_dir; skipping writer")
        return None
    path = Path(log_dir)
    if run_name:
        path = path / run_name
    path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(path))
    logger.info("tensorboard writing to {}", path.resolve())
    logger.debug("open tensorboard with: tensorboard --logdir {}", Path(log_dir).resolve())
    return writer


def log_scalars(writer: SummaryWriter | None, step: int, scalars: dict[str, float]) -> None:
    if writer is None:
        return
    for name, value in scalars.items():
        writer.add_scalar(name, value, step)


def log_hparams(writer: SummaryWriter | None, hparams: dict[str, Any], metrics: dict[str, float] | None = None) -> None:
    if writer is None:
        return
    clean = {}
    for k, v in hparams.items():
        if isinstance(v, (int, float, str, bool)):
            clean[k] = v
        elif v is None:
            clean[k] = "none"
        else:
            clean[k] = str(v)
    writer.add_hparams(clean, metrics or {"hparam/placeholder": 0.0})
