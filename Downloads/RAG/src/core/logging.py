"""Loguru-based logging with per-run log files. Never logs secrets."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_CONFIGURED = False


def setup_logging(log_dir: str | Path, level: str = "INFO") -> None:
    """Configure stderr + rotating file sinks once per process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stderr, level=level, format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")
    logger.add(
        log_path / "ragbench_{time:YYYY-MM-DD}.log",
        level=level,
        rotation="10 MB",
        retention="14 days",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {name}:{line} | {message}",
    )
    _CONFIGURED = True


def get_logger(name: str):
    return logger.bind(module=name)
