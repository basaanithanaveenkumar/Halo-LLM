"""Loguru logging for hale-llm. Call setup_logging() once from CLI entry points."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_CONFIGURED = False


def setup_logging(
    level: str = "INFO",
    log_file: str | None = "logs/hale_llm.log",
    *,
    force: bool = False,
) -> None:
    """Install stderr + optional rotating file sinks.

    `level` is the console level (DEBUG, INFO, WARNING, ERROR).
    The file sink always records DEBUG and above when enabled.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return
    level = level.upper()
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(path),
            level="DEBUG",
            rotation="10 MB",
            retention="7 days",
            encoding="utf-8",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        )
        logger.debug("file logging enabled at {}", path.resolve())
    _CONFIGURED = True
    logger.info("logging initialized (console={})", level)
