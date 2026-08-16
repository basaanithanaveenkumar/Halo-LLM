from __future__ import annotations

import argparse

import ha_llm.models  # noqa: F401
from loguru import logger

from ha_llm.config.experiment import apply_experiment_layout
from ha_llm.config.schema import load_config
from ha_llm.inference import load_session
from ha_llm.utils.logging import setup_logging


def main() -> None:
    p = argparse.ArgumentParser(prog="ha-llm-sample")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--experiment", default=None, help="Run folder under data/experiments/<variant>/")
    p.add_argument("--prompt", default=None)
    p.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args()
    try:
        cfg = load_config(args.config)
        if args.log_level:
            cfg.logging.level = args.log_level
        apply_experiment_layout(cfg, create=False, name=args.experiment)
        setup_logging(level=cfg.logging.level, log_file=cfg.logging.log_file, force=True)
        session = load_session(cfg, args.checkpoint)
        result = session.generate(args.prompt)
        print(result.text)
    except Exception:
        logger.exception("ha-llm-sample failed")
        raise


if __name__ == "__main__":
    main()
