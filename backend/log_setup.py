"""Logging setup for Diffucore UI: run-id stamping, one timestamped format for
every ``diffucore.*`` logger, and an optional rotating ``--log-file``. Called
from ``app.py`` before ``server`` is imported. Secrets stay on ``print()``.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import secrets
import sys
from pathlib import Path
from typing import Optional

_RUN_ID = ""


def run_id() -> str:
    return _RUN_ID


def configure(
    *,
    log_file: Optional[str] = None,
    level: str = "INFO",
    max_bytes: int = 5 * 1024 * 1024,
    backups: int = 3,
) -> str:
    """Configure root logging: stderr always, plus a rotating chmod-600 file
    when ``log_file`` is given. Returns the per-process run-id.
    """
    global _RUN_ID
    _RUN_ID = secrets.token_hex(4)

    numeric = getattr(logging, level.upper(), logging.INFO)
    fmt = logging.Formatter(
        f"%(asctime)s [{_RUN_ID}] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    # Drop handlers from a previous configure so lines aren't emitted twice.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(numeric)

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    if log_file:
        path = Path(log_file)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(
                path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8",
            )
            fh.setFormatter(fmt)
            root.addHandler(fh)
            # The log names models and paths; keep it owner-only.
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
        except OSError as e:
            # A bad log path shouldn't kill the server.
            root.warning("could not open log file %s: %s (logging to stderr only)",
                         path, e)

    logging.getLogger("diffucore").info(
        "logging configured (run_id=%s, level=%s, file=%s)",
        _RUN_ID, level.upper(), log_file or "none",
    )
    return _RUN_ID
