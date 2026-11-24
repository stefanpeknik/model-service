"""Capture what a process's logging configuration writes to stdout."""

from __future__ import annotations

import io
import logging
from contextlib import redirect_stdout
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import unittest
    from collections.abc import Callable


# The loggers the control plane and replica configurations touch.
_CONFIGURED = ("", "model_service", "uvicorn", "uvicorn.access")


def capture_logs(test: unittest.TestCase, configure: Callable[[], None]) -> io.StringIO:
    """Run a logging configuration with stdout going to a buffer.

    The loggers are restored when the test ends.

    Args:
        test: The test whose cleanup restores the loggers.
        configure: Sets up logging as the process does.

    Returns:
        The buffer the configured handlers write to.
    """
    saved = [
        (logger, logger.handlers[:], logger.level, logger.propagate)
        for logger in map(logging.getLogger, _CONFIGURED)
    ]

    def restore() -> None:
        for logger, handlers, level, propagate in saved:
            logger.handlers = handlers
            logger.level = level
            logger.propagate = propagate

    test.addCleanup(restore)
    stream = io.StringIO()
    with redirect_stdout(stream):
        configure()
    return stream
