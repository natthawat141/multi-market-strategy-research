"""Shared logging setup for FRTBOT scripts and notebooks.

Rules to enforce loudly (per AGENTS.md): ambiguous currency, timezone,
duplicate keys, non-monotonic timestamps, or invalid adjustment metadata must
raise, not warn. Logging here is for pipeline progress and data-quality
findings that are not themselves fatal (e.g. a market falling back to proxy
mode).
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module logger, configuring the root handler once per process."""
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
            stream=sys.stdout,
        )
        _CONFIGURED = True
    return logging.getLogger(name)
