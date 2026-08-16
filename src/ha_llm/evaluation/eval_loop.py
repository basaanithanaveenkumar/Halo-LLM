"""Eval loop: run registered metrics over a loader. Metric classes live in ha_llm.metrics."""

from __future__ import annotations

import torch
from loguru import logger

from ha_llm.config.schema import RunConfig
from ha_llm.core.registry import get_loss, instantiate_metrics
from ha_llm.core.tensors import move_batch_to_device
from ha_llm.metrics.base import MetricContext
from ha_llm.metrics import general as _shared_metrics  # noqa: F401
from ha_llm.utils.device import get_device


class Evaluator:
    def __init__(self, cfg: RunConfig, device: str | None = None) -> None:
        self.cfg = cfg
        self.device = device or get_device(cfg.device)
        self.loss_fn = get_loss(cfg.variant)

    @torch.no_grad()
    def run(self, model, loader) -> dict[str, float]:
        metric_objs = instantiate_metrics(list(self.cfg.eval.metrics), self.cfg.variant)
        if not metric_objs:
            logger.warning("no metrics instantiated for variant={}", self.cfg.variant)
            return {}
        for m in metric_objs:
            m.reset()
        model.eval()
        n = 0
        max_batches = self.cfg.eval.max_batches
        logger.info(
            "evaluating variant={} metrics={} batches={} max_batches={}",
            self.cfg.variant,
            [getattr(m, "metric_name", type(m).__name__) for m in metric_objs],
            len(loader),
            max_batches,
        )
        for batch in loader:
            if max_batches is not None and n >= max_batches:
                break
            batch = move_batch_to_device(batch, self.device)
            loss = self.loss_fn(model, batch)
            ctx = MetricContext(model=model, batch=batch, loss=loss, variant=self.cfg.variant)
            for m in metric_objs:
                m.update(ctx)
            n += 1
        model.train()
        if n == 0:
            logger.warning("eval loader was empty")
        out: dict[str, float] = {}
        for m in metric_objs:
            out.update(m.compute())
        logger.info("eval over {} batches: {}", n, {k: round(v, 4) for k, v in out.items()})
        return out


def evaluate(model, cfg, loader) -> dict[str, float]:
    return Evaluator(cfg).run(model, loader)
