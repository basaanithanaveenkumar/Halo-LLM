from __future__ import annotations

import argparse

import ha_llm.models  # noqa: F401
from loguru import logger

from ha_llm.config.experiment import apply_experiment_layout
from ha_llm.config.schema import load_config
from ha_llm.dataloader.dataset import make_dataloader
from ha_llm.evaluation.eval_loop import evaluate
from ha_llm.inference import load_session
from ha_llm.utils.logging import setup_logging
from ha_llm.utils.tensorboard import log_scalars, make_writer


def main() -> None:
    p = argparse.ArgumentParser(prog="ha-llm-eval")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--experiment", default=None, help="Run folder under data/experiments/<variant>/")
    p.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument(
        "--metrics",
        default=None,
        help="Comma-separated metric names (overrides eval.metrics)",
    )
    args = p.parse_args()
    try:
        cfg = load_config(args.config)
        if args.log_level:
            cfg.logging.level = args.log_level
        if args.metrics:
            cfg.eval.metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
        apply_experiment_layout(cfg, create=False, name=args.experiment)
        setup_logging(level=cfg.logging.level, log_file=cfg.logging.log_file, force=True)
        session = load_session(cfg, args.checkpoint, require_checkpoint=True)
        loader = make_dataloader(cfg, split="validation")
        metrics = evaluate(session.model, cfg, loader)
        writer = make_writer(
            cfg.logging.tensorboard_dir,
            enabled=cfg.logging.tensorboard,
            run_name=None if cfg.experiment.enabled else f"{cfg.variant}_eval",
        )
        log_scalars(writer, 0, {f"eval/{k}": v for k, v in metrics.items()})
        if writer is not None:
            writer.close()
        for name, value in metrics.items():
            logger.info("eval {}: {:.4f}", name, value)
            print(f"{name}: {value:.4f}")
    except Exception:
        logger.exception("ha-llm-eval failed")
        raise


if __name__ == "__main__":
    main()
