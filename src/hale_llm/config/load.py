"""Load and inherit YAML into a validated RunConfig."""

from pathlib import Path

import yaml
from loguru import logger

from hale_llm.config.merge import deep_merge
from hale_llm.config.run import RunConfig


def load_config(path: str | Path) -> RunConfig:
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
            parent_raw = deep_merge(grand_raw, parent_raw)
        raw = deep_merge(parent_raw, raw)
    try:
        cfg = RunConfig.model_validate(raw)
    except Exception:
        logger.exception("invalid config {}", path)
        raise
    cfg.experiment.source_yaml = str(path)
    logger.info("loaded config variant={} from {}", cfg.variant, path)
    return cfg
