"""Central logging configuration for FL-IDS.

Every module should use `logging.getLogger(__name__)` and call
`setup_logging()` once from an entrypoint (orchestration scripts, tests)
rather than using bare `print()` statements, so run output stays usable
and its verbosity is configurable.
"""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(level: str = "INFO", log_dir: str | None = None, log_file: str = "fl_ids.log") -> None:
    """Configure the root logger with a console handler and optional file handler.

    Args:
        level: Logging level name (e.g. "DEBUG", "INFO", "WARNING").
        log_dir: If given, also write logs to `{log_dir}/{log_file}`, creating
            the directory if needed.
        log_file: File name used when `log_dir` is given.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(Path(log_dir) / log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
