"""Thin CLI: load via inference → write a generation GIF."""

from __future__ import annotations

import argparse

import hale_llm.models  # noqa: F401
from loguru import logger

from hale_llm.config.experiment import apply_experiment_layout
from hale_llm.config.schema import load_config
from hale_llm.inference import load_session
from hale_llm.utils.logging import setup_logging
from hale_llm.visualization.run import run_visualization


def main() -> None:
    p = argparse.ArgumentParser(prog="hale-llm-viz")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--experiment", default=None, help="Run folder under data/experiments/<variant>/")
    p.add_argument("--prompt", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args()
    try:
        cfg = load_config(args.config)
        if args.log_level:
            cfg.logging.level = args.log_level
        if args.prompt:
            cfg.sample.prompt = args.prompt
            cfg.viz.prompt_source = "config"
        apply_experiment_layout(cfg, create=False, name=args.experiment)
        setup_logging(level=cfg.logging.level, log_file=cfg.logging.log_file, force=True)
        session = load_session(cfg, args.checkpoint)
        default_name = f"{cfg.viz.output_dir.rstrip('/')}/reveal.gif"
        out = args.output or default_name
        logger.info("visualize variant={} prompt={!r}", cfg.variant, cfg.sample.prompt)
        run_visualization(session.model, session.tokenizer, cfg, session.device, out)
    except Exception:
        logger.exception("hale-llm-viz failed")
        raise


if __name__ == "__main__":
    main()
