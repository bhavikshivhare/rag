"""LangSmith tracing setup.

LangSmith is configured entirely through environment variables, so this module only
translates the YAML ``tracing`` section into them. The API key is never read from a
config file and never logged: without ``LANGCHAIN_API_KEY`` (or whatever
``tracing.api_key_env`` names) tracing stays off and the run continues untraced.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from src.core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.core.config import Settings

log = get_logger(__name__)

ENABLED_VAR = "LANGCHAIN_TRACING_V2"


def configure_tracing(settings: "Settings") -> bool:
    """Enable LangSmith tracing when configured *and* a key is present. Idempotent."""
    cfg = settings.tracing
    if not cfg.enabled:
        os.environ.setdefault(ENABLED_VAR, "false")
        return False

    api_key = os.environ.get(cfg.api_key_env, "").strip()
    if not api_key:
        log.warning(
            "tracing.enabled is set but {} is empty; running without LangSmith tracing",
            cfg.api_key_env,
        )
        os.environ[ENABLED_VAR] = "false"
        return False

    os.environ[ENABLED_VAR] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = cfg.project
    os.environ["LANGCHAIN_ENDPOINT"] = cfg.endpoint
    log.info("LangSmith tracing enabled (project={})", cfg.project)
    return True
