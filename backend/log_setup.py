"""Structured logging setup for Diffucore UI.

Centralises what used to be a mix of ``print()`` (startup, share) and ad-hoc
``logging`` (extensions only): one configuration point, run-id stamping, an
optional ``--log-file`` with size rotation, and a single timestamped format
across every ``diffucore.*`` logger. Called once from ``app.py`` *before*
``server`` is imported so the module-level log calls in ``server.py`` are
captured.

Token / share-URL secrets stay on ``print()`` (stdout only) so they never land
in the log file — only operational messages go through here.
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
    """Configure root + ``diffucore`` logging.

    Always attaches a stream handler (stderr) so console behaviour is preserved;
    when ``log_file`` is given, also attaches a rotating file handler (``chmod
    600`` on POSIX) so a long-running server doesn't grow one file unbounded.
    Returns the per-process run-id (also stamped into every log line via the
    format and into the startup banner).
    """
    global _RUN_ID
    _RUN_ID = secrets.token_hex(4)

    numeric = getattr(logging, level.upper(), logging.INFO)
    fmt = logging.Formatter(
        f"%(asctime)s [{_RUN_ID}] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    # Drop any prior handlers from a re-configure (e.g. test re-import) so we
    # don't double-emit lines.
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
            # Owner-only on POSIX so the log (which names models/paths) isn't
            # world-readable on a shared host. Windows ignores chmod.
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
        except OSError as e:
            # A bad log path shouldn't kill the server — fall back to stderr.
            root.warning("could not open log file %s: %s (logging to stderr only)",
                         path, e)

    logging.getLogger("diffucore").info(
        "logging configured (run_id=%s, level=%s, file=%s)",
        _RUN_ID, level.upper(), log_file or "—",
    )
    return _RUN_ID
